# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Implementation of MASt3R training losses
# --------------------------------------------------------

# Modifications Copyright (C) <Alibaba Group>
# Changes: loss function update
# This is an adaptation and is distributed under the same license (CC BY-NC-SA 4.0).
# SPDX-License-Identifier: CC-BY-NC-SA-4.0(non-commercial use only)
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.losses import BaseCriterion, Criterion, MultiLoss, Sum, ConfLoss
from dust3r.losses import Regr3D as Regr3D_dust3r
from dust3r.utils.geometry import (geotrf, inv, normalize_pointcloud)
from dust3r.inference import get_pred_pts3d
from dust3r.utils.geometry import get_joint_pointcloud_depth, get_joint_pointcloud_center_scale
from mast3r.ssim import ssim, SSIM


def apply_log_to_norm(xyz):
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    xyz = xyz * torch.log1p(d)
    return xyz

def normalized_depth_scale_and_shift(prediction, target, mask):
    """
    More info here: https://arxiv.org/pdf/2206.00665.pdf supplementary section A2 Depth Consistency Loss
    This function computes scale/shift required to normalizes predicted depth map,
    to allow for using normalized depth maps as input from monocular depth estimation networks.
    These networks are trained such that they predict normalized depth maps.

    Solves for scale/shift using a least squares approach with a closed form solution:
    Based on:
    https://github.com/autonomousvision/monosdf/blob/d9619e948bf3d85c6adec1a643f679e2e8e84d4b/code/model/loss.py#L7
    Args:
        prediction: predicted depth map
        target: ground truth depth map
        mask: mask of valid pixels
    Returns:
        scale and shift for depth prediction
    """
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    scale = torch.zeros_like(b_0)
    shift = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return scale, shift

'''
Regr3D(L21, norm_mode='avg_dis', align_pts=False, sky_loss_value=0)
'''

class Regr3D (Regr3D_dust3r):
    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False, opt_fit_gt=False,
                 sky_loss_value=2, max_metric_scale=False, loss_in_log=False, align_pts=True, 
                 add_align_depth_loss=False):
        self.loss_in_log = loss_in_log
        if norm_mode.startswith('?'):
            # do no norm pts from metric scale datasets
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
        super().__init__(criterion, self.norm_mode, gt_scale)

        self.sky_loss_value = sky_loss_value
        self.max_metric_scale = max_metric_scale
        self.align_pts = align_pts
        self.add_align_depth_loss = add_align_depth_loss

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gt1['camera_pose'])
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3
        if self.align_pts:
            gt_pts2 = geotrf(in_camera1, gt2['pts3d'])  # B,H,W,3
        else:
            in_camera2 = inv(gt2['camera_pose'])
            gt_pts2 = geotrf(in_camera2, gt2['pts3d'])

        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)

        if self.loss_in_log == 'before':
            # this only make sense when depth_mode == 'linear'
            gt_pts1 = apply_log_to_norm(gt_pts1)
            gt_pts2 = apply_log_to_norm(gt_pts2)

        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False).clone()
        pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True).clone()

        if not self.norm_all: # metric scale depth do not norm
            if self.max_metric_scale:
                B = valid1.shape[0]
                # valid1: B, H, W
                # torch.linalg.norm(gt_pts1, dim=-1) -> B, H, W
                # dist1_to_cam1 -> reshape to B, H*W
                dist1_to_cam1 = torch.where(valid1, torch.linalg.norm(gt_pts1, dim=-1), 0).view(B, -1)
                dist2_to_cam1 = torch.where(valid2, torch.linalg.norm(gt_pts2, dim=-1), 0).view(B, -1)

                # is_metric_scale: B
                # dist1_to_cam1.max(dim=-1).values -> B
                gt1['is_metric_scale'] = gt1['is_metric_scale'] \
                    & (dist1_to_cam1.max(dim=-1).values < self.max_metric_scale) \
                    & (dist2_to_cam1.max(dim=-1).values < self.max_metric_scale)
                gt2['is_metric_scale'] = gt1['is_metric_scale']

            mask = ~gt1['is_metric_scale']
        else:
            mask = torch.ones_like(gt1['is_metric_scale'])
        # normalize 3d points
        if self.norm_mode and mask.any():
            pr_pts1[mask], pr_pts2[mask] = normalize_pointcloud(pr_pts1[mask], pr_pts2[mask], self.norm_mode,
                                                                valid1[mask], valid2[mask])

        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2, norm_factor = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode,
                                                                 valid1, valid2, ret_factor=True)
            # apply the same normalization to prediction
            pr_pts1[~mask] = pr_pts1[~mask] / norm_factor[~mask]
            pr_pts2[~mask] = pr_pts2[~mask] / norm_factor[~mask]

        # return sky segmentation, making sure they don't include any labelled 3d points
        sky1 = gt1['sky_mask'] & (~valid1)
        sky2 = gt2['sky_mask'] & (~valid2)
        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, sky1, sky2, {}
    
    def get_all_pts3d_align(self, gt1, gt2, pred1, pred2):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gt1['camera_pose'])
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3
        if self.align_pts:
            gt_pts2 = geotrf(in_camera1, gt2['pts3d'])  # B,H,W,3
        else:
            in_camera2 = inv(gt2['camera_pose'])
            gt_pts2 = geotrf(in_camera2, gt2['pts3d'])

        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()

        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False).clone()
        pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True).clone()
        
        # align pointcloud
        scale1, shift1 = normalized_depth_scale_and_shift(
            prediction=pr_pts1[..., -1], 
            target=gt_pts1[..., -1], 
            mask=valid1)
        scale1 = scale1.reshape(-1, 1, 1, 1)
        shift1 = shift1.reshape(-1, 1, 1, 1)
        pr_pts1 = pr_pts1 * scale1 + shift1
        
        scale2, shift2 = normalized_depth_scale_and_shift(
            prediction=pr_pts2[..., -1], 
            target=gt_pts2[..., -1], 
            mask=valid2)
        scale2 = scale2.reshape(-1, 1, 1, 1)
        shift2 = shift2.reshape(-1, 1, 1, 1)
        pr_pts2 = pr_pts2 * scale2 + shift2

        # return sky segmentation, making sure they don't include any labelled 3d points
        sky1 = gt1['sky_mask'] & (~valid1)
        sky2 = gt2['sky_mask'] & (~valid2)
        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, sky1, sky2, {}

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
            self.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)
        
        # gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
        #     self.get_all_pts3d_align(gt1, gt2, pred1, pred2, **kw)

        if self.sky_loss_value > 0:
            assert self.criterion.reduction == 'none', 'sky_loss_value should be 0 if no conf loss'
            # add the sky pixel as "valid" pixels...
            mask1 = mask1 | sky1
            mask2 = mask2 | sky2

        # loss on img1 side
        pred_pts1 = pred_pts1[mask1]
        gt_pts1 = gt_pts1[mask1]
        if self.loss_in_log and self.loss_in_log != 'before':
            # this only make sense when depth_mode == 'exp'
            pred_pts1 = apply_log_to_norm(pred_pts1)
            gt_pts1 = apply_log_to_norm(gt_pts1)
        l1 = self.criterion(pred_pts1, gt_pts1)

        # loss on gt2 side
        pred_pts2 = pred_pts2[mask2]
        gt_pts2 = gt_pts2[mask2]
        if self.loss_in_log and self.loss_in_log != 'before':
            pred_pts2 = apply_log_to_norm(pred_pts2)
            gt_pts2 = apply_log_to_norm(gt_pts2)
        l2 = self.criterion(pred_pts2, gt_pts2)

        if self.sky_loss_value > 0:
            assert self.criterion.reduction == 'none', 'sky_loss_value should be 0 if no conf loss'
            # ... but force the loss to be high there
            l1 = torch.where(sky1[mask1], self.sky_loss_value, l1)
            l2 = torch.where(sky2[mask2], self.sky_loss_value, l2)
        self_name = type(self).__name__
        details = {self_name + '_pts3d_1': float(l1.mean()), self_name + '_pts3d_2': float(l2.mean())}
        
        return Sum((l1, mask1), (l2, mask2)), (details | monitoring)
    

class AlignDepthLoss(MultiLoss):
    def __init__(self, align_pts):
        super().__init__()
        self.align_pts = align_pts

    def get_name(self):
        return f'AlignDepthLoss'
    
    def compute_align_depth_loss(self, gt1, gt2, pred1, pred2):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gt1['camera_pose'])
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3
        if self.align_pts:
            gt_pts2 = geotrf(in_camera1, gt2['pts3d'])  # B,H,W,3
        else:
            in_camera2 = inv(gt2['camera_pose'])
            gt_pts2 = geotrf(in_camera2, gt2['pts3d'])

        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()

        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False).clone()
        pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True).clone()
        
        # align pointcloud
        scale1, shift1 = normalized_depth_scale_and_shift(
            prediction=pr_pts1[..., -1], 
            target=gt_pts1[..., -1], 
            mask=valid1)
        scale1 = scale1.reshape(-1, 1, 1, 1)
        shift1 = shift1.reshape(-1, 1, 1, 1)
        pre_depth1 = pr_pts1[..., -1] * scale1[..., 0] + shift1[..., 0]
        
        scale2, shift2 = normalized_depth_scale_and_shift(
            prediction=pr_pts2[..., -1], 
            target=gt_pts2[..., -1], 
            mask=valid2)
        scale2 = scale2.reshape(-1, 1, 1, 1)
        shift2 = shift2.reshape(-1, 1, 1, 1)
        pre_depth2 = pr_pts2[..., -1] * scale2[..., 0] + shift2[..., 0]
        
        # compute loss L1
        loss1 = torch.abs(pre_depth1 - gt_pts1[..., -1])[valid1].mean()
        loss2 = torch.abs(pre_depth2 - gt_pts2[..., -1])[valid2].mean()
        
        return loss1, loss2
    
    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        loss1, loss2 = self.compute_align_depth_loss(gt1, gt2, pred1, pred2)
        loss_count = loss1 + loss2
        
        details = {"align_depth_loss_1": float(loss1), 
                   "align_depth_loss_2": float(loss2),
                   "align_depth_loss_count": float(loss_count)}
        
        return loss_count, details
    
    
class RawDepthLoss(MultiLoss):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return f'RawDepthLoss'
    
    def compute_depth_loss(self, gt1, gt2, pred1, pred2):
        in_camera1 = inv(gt1['camera_pose'])
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3
        gt_depth1 = gt_pts1[..., -1:]
        
        in_camera2 = inv(gt2['camera_pose'])
        gt_pts2 = geotrf(in_camera2, gt2['pts3d'])  # B,H,W,3
        gt_depth2 = gt_pts2[..., -1:]

        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()

        pred_depth1 = pred1["depth"]
        pred_depth2 = pred2["depth"]
        
        # align pointcloud
        scale1, shift1 = normalized_depth_scale_and_shift(
            prediction=pred_depth1[..., 0], 
            target=gt_depth1[..., 0], 
            mask=valid1)
        scale1 = scale1.reshape(-1, 1, 1, 1)
        shift1 = shift1.reshape(-1, 1, 1, 1)
        pre_depth1 = pred_depth1[..., -1] * scale1[..., 0] + shift1[..., 0]
        
        scale2, shift2 = normalized_depth_scale_and_shift(
            prediction=pred_depth2[..., 0], 
            target=gt_depth2[..., 0], 
            mask=valid2)
        scale2 = scale2.reshape(-1, 1, 1, 1)
        shift2 = shift2.reshape(-1, 1, 1, 1)
        pre_depth2 = pred_depth2[..., -1] * scale2[..., 0] + shift2[..., 0]
        
        # compute loss L1
        loss1 = torch.abs(pre_depth1 - gt_pts1[..., -1])[valid1].mean()
        loss2 = torch.abs(pre_depth2 - gt_pts2[..., -1])[valid2].mean()
        
        return loss1, loss2
    
    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        loss1, loss2 = self.compute_depth_loss(gt1, gt2, pred1, pred2)
        loss_count = loss1 + loss2
        
        details = {"raw_depth_loss_1": float(loss1), 
                   "raw_depth_loss_2": float(loss2),
                   "raw_depth_loss_count": float(loss_count)}
        
        return loss_count, details

class SegmentationLoss(MultiLoss):
    def __init__(self):
        super().__init__()
        self.seg_criterion = nn.CrossEntropyLoss()

    def get_name(self):
        return f'SegmentationLoss'

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        seg_label1 = gt1["seg_label"].long()
        seg_label2 = gt2["seg_label"].long()

        seg_pred1 = pred1["seg_prob"] # B,C,H,W
        seg_pred2 = pred2["seg_prob"] # B,C,H,W

        loss = self.seg_criterion(seg_pred1, seg_label1) + self.seg_criterion(seg_pred2, seg_label2)

        details = {"SegmentationLoss": float(loss)}
        
        return loss, details


class Regr3D_ShiftInv (Regr3D):
    """ Same than Regr3D but invariant to depth shift.
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute unnormalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # compute median depth
        gt_z1, gt_z2 = gt_pts1[..., 2], gt_pts2[..., 2]
        pred_z1, pred_z2 = pred_pts1[..., 2], pred_pts2[..., 2]
        gt_shift_z = get_joint_pointcloud_depth(gt_z1, gt_z2, mask1, mask2)[:, None, None]
        pred_shift_z = get_joint_pointcloud_depth(pred_z1, pred_z2, mask1, mask2)[:, None, None]

        # subtract the median depth
        gt_z1 -= gt_shift_z
        gt_z2 -= gt_shift_z
        pred_z1 -= pred_shift_z
        pred_z2 -= pred_shift_z

        # monitoring = dict(monitoring, gt_shift_z=gt_shift_z.mean().detach(), pred_shift_z=pred_shift_z.mean().detach())
        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring


class Regr3D_ScaleInv (Regr3D):
    """ Same than Regr3D but invariant to depth scale.
        if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute depth-normalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # measure scene scale
        _, gt_scale = get_joint_pointcloud_center_scale(gt_pts1, gt_pts2, mask1, mask2)
        _, pred_scale = get_joint_pointcloud_center_scale(pred_pts1, pred_pts2, mask1, mask2)

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pred_pts1 *= gt_scale / pred_scale
            pred_pts2 *= gt_scale / pred_scale
        else:
            gt_pts1 /= gt_scale
            gt_pts2 /= gt_scale
            pred_pts1 /= pred_scale
            pred_pts2 /= pred_scale

        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring


class Regr3D_ScaleShiftInv (Regr3D_ScaleInv, Regr3D_ShiftInv):
    # calls Regr3D_ShiftInv first, then Regr3D_ScaleInv
    pass


def get_similarities(desc1, desc2, euc=False):
    if euc:  # euclidean distance in same range than similarities
        dists = (desc1[:, :, None] - desc2[:, None]).norm(dim=-1)
        sim = 1 / (1 + dists)
    else:
        # Compute similarities
        sim = desc1 @ desc2.transpose(-2, -1)
    return sim


class MatchingCriterion(BaseCriterion):
    def __init__(self, reduction='mean', fp=torch.float32):
        super().__init__(reduction)
        self.fp = fp

    def forward(self, a, b, valid_matches=None, euc=False):
        assert a.ndim >= 2 and 1 <= a.shape[-1], f'Bad shape = {a.shape}'
        dist = self.loss(a.to(self.fp), b.to(self.fp), valid_matches, euc=euc)
        # one dimension less or reduction to single value
        assert (valid_matches is None and dist.ndim == a.ndim -
                1) or self.reduction in ['mean', 'sum', '1-mean', 'none']
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        if self.reduction == '1-mean':
            return 1. - dist.mean() if dist.numel() > 0 else dist.new_ones(())
        raise ValueError(f'bad {self.reduction=} mode')

    def loss(self, a, b, valid_matches=None):
        raise NotImplementedError


class InfoNCE(MatchingCriterion):
    def __init__(self, temperature=0.07, eps=1e-8, mode='all', **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.eps = eps
        assert mode in ['all', 'proper', 'dual']
        self.mode = mode

    def loss(self, desc1, desc2, valid_matches=None, euc=False):
        # valid positives are along diagonals
        B, N, D = desc1.shape
        B2, N2, D2 = desc2.shape
        assert B == B2 and D == D2
        if valid_matches is None:
            valid_matches = torch.ones([B, N], dtype=bool)
        # torch.all(valid_matches.sum(dim=-1) > 0) some pairs have no matches????
        assert valid_matches.shape == torch.Size([B, N])# and valid_matches.sum() > 0
        assert valid_matches.sum() > 0

        # Tempered similarities
        sim = get_similarities(desc1, desc2, euc) / self.temperature
        sim[sim.isnan()] = -torch.inf  # ignore nans
        # Softmax of positives with temperature
        sim = sim.exp_()  # save peak memory
        positives = sim.diagonal(dim1=-2, dim2=-1)

        # Loss
        if self.mode == 'all':            # Previous InfoNCE
            loss = -torch.log((positives / sim.sum(dim=-1).sum(dim=-1, keepdim=True)).clip(self.eps))
        elif self.mode == 'proper':  # Proper InfoNCE
            loss = -(torch.log((positives / sim.sum(dim=-2)).clip(self.eps)) +
                     torch.log((positives / sim.sum(dim=-1)).clip(self.eps)))
        elif self.mode == 'dual':  # Dual Softmax
            loss = -(torch.log((positives**2 / sim.sum(dim=-1) / sim.sum(dim=-2)).clip(self.eps)))
        else:
            raise ValueError("This should not happen...")
        return loss[valid_matches]


class APLoss (MatchingCriterion):
    """ AP loss.

        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}

        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """

    def __init__(self, nq='torch', min=0, max=1, euc=False, **kw):
        super().__init__(**kw)
        # Exact/True AP loss (not differentiable)
        if nq == 0:
            nq = 'sklearn'  # special case
        try:
            self.compute_AP = eval('self.compute_true_AP_' + nq)
        except:
            raise ValueError("Unknown mode %s for AP loss" % nq)

    @staticmethod
    def compute_true_AP_sklearn(scores, labels):
        def compute_AP(label, score):
            return average_precision_score(label, score)

        aps = scores.new_zeros((scores.shape[0], scores.shape[1]))
        label_np = labels.cpu().numpy().astype(bool)
        scores_np = scores.cpu().numpy()
        for bi in range(scores_np.shape[0]):
            for i in range(scores_np.shape[1]):
                labels = label_np[bi, i, :]
                if labels.sum() < 1:
                    continue
                aps[bi, i] = compute_AP(labels, scores_np[bi, i, :])
        return aps

    @staticmethod
    def compute_true_AP_torch(scores, labels):
        assert scores.shape == labels.shape
        B, N, M = labels.shape
        dev = labels.device
        with torch.no_grad():
            # sort scores
            _, order = scores.sort(dim=-1, descending=True)
            # sort labels accordingly
            labels = labels[torch.arange(B, device=dev)[:, None, None].expand(order.shape),
                            torch.arange(N, device=dev)[None, :, None].expand(order.shape),
                            order]
            # compute number of positives per query
            npos = labels.sum(dim=-1)
            assert torch.all(torch.isclose(npos, npos[0, 0])
                             ), "only implemented for constant number of positives per query"
            npos = int(npos[0, 0])
            # compute precision at each recall point
            posrank = labels.nonzero()[:, -1].view(B, N, npos)
            recall = torch.arange(1, 1 + npos, dtype=torch.float32, device=dev)[None, None, :].expand(B, N, npos)
            precision = recall / (1 + posrank).float()
            # average precision values at all recall points
            aps = precision.mean(dim=-1)

        return aps

    def loss(self, desc1, desc2, valid_matches=None, euc=False):  # if matches is None, positives are the diagonal
        B, N1, D = desc1.shape
        B2, N2, D2 = desc2.shape
        assert B == B2 and D == D2

        scores = get_similarities(desc1, desc2, euc) # scores higher -> matching better, B,N1,N2

        labels = torch.zeros([B, N1, N2], dtype=scores.dtype, device=scores.device)

        # allow all diagonal positives and only mask afterwards
        labels.diagonal(dim1=-2, dim2=-1)[...] = 1.
        apscore = self.compute_AP(scores, labels)
        if valid_matches is not None:
            apscore = apscore[valid_matches]
        return apscore


class MatchingLoss (Criterion, MultiLoss):
    """ 
    Matching loss per image 
    only compare pixels inside an image but not in the whole batch as what would be done usually
    """

    def __init__(self, criterion, withconf=False, use_pts3d=False, negatives_padding=0, blocksize=4096):
        super().__init__(criterion)
        self.negatives_padding = negatives_padding
        self.use_pts3d = use_pts3d
        self.blocksize = blocksize
        self.withconf = withconf

    def add_negatives(self, outdesc2, desc2, batchid, x2, y2):
        if self.negatives_padding:
            B, H, W, D = desc2.shape
            negatives = torch.ones([B, H, W], device=desc2.device, dtype=bool)
            negatives[batchid, y2, x2] = False
            sel = negatives & (negatives.view([B, -1]).cumsum(dim=-1).view(B, H, W)
                               <= self.negatives_padding)  # take the N-first negatives
            outdesc2 = torch.cat([outdesc2, desc2[sel].view([B, -1, D])], dim=1)
        return outdesc2

    def get_confs(self, pred1, pred2, sel1, sel2):
        if self.withconf:
            if self.use_pts3d:
                outconfs1 = pred1['conf'][sel1]
                outconfs2 = pred2['conf'][sel2]
            else:
                outconfs1 = pred1['desc_conf'][sel1]
                outconfs2 = pred2['desc_conf'][sel2]
        else:
            outconfs1 = outconfs2 = None
        return outconfs1, outconfs2

    def get_descs(self, pred1, pred2):
        if self.use_pts3d:
            desc1, desc2 = pred1['pts3d'], pred2['pts3d_in_other_view']
        else:
            desc1, desc2 = pred1['desc'], pred2['desc']
        return desc1, desc2

    def get_matching_descs(self, gt1, gt2, pred1, pred2, **kw):
        outdesc1 = outdesc2 = outconfs1 = outconfs2 = None
        # Recover descs, GT corres and valid mask
        desc1, desc2 = self.get_descs(pred1, pred2)

        (x1, y1), (x2, y2) = gt1['corres'].unbind(-1), gt2['corres'].unbind(-1)
        valid_matches = gt1['valid_corres']

        # Select descs that have GT matches
        B, N = x1.shape
        batchid = torch.arange(B)[:, None].repeat(1, N)  # B, N
        outdesc1, outdesc2 = desc1[batchid, y1, x1], desc2[batchid, y2, x2]  # B, N, D

        # Padd with unused negatives
        outdesc2 = self.add_negatives(outdesc2, desc2, batchid, x2, y2)

        # Gather confs if needed
        sel1 = batchid, y1, x1
        sel2 = batchid, y2, x2
        outconfs1, outconfs2 = self.get_confs(pred1, pred2, sel1, sel2)

        return outdesc1, outdesc2, outconfs1, outconfs2, valid_matches, {'use_euclidean_dist': self.use_pts3d}

    def blockwise_criterion(self, descs1, descs2, confs1, confs2, valid_matches, euc, rng=np.random, shuffle=True):
        loss = None
        details = {}
        B, N, D = descs1.shape

        if N <= self.blocksize:  # Blocks are larger than provided descs, compute regular loss
            loss = self.criterion(descs1, descs2, valid_matches, euc=euc) # info_NCE_loss / APLoss
        else:  # Compute criterion on the blockdiagonal only, after shuffling
            # Shuffle if necessary
            matches_perm = slice(None)
            if shuffle:
                matches_perm = np.stack([rng.choice(range(N), size=N, replace=False) for _ in range(B)])
                batchid = torch.tile(torch.arange(B), (N, 1)).T
                matches_perm = batchid, matches_perm

            descs1 = descs1[matches_perm]
            descs2 = descs2[matches_perm]
            valid_matches = valid_matches[matches_perm]

            assert N % self.blocksize == 0, "Error, can't chunk block-diagonal, please check blocksize"
            n_chunks = N // self.blocksize
            descs1 = descs1.reshape([B * n_chunks, self.blocksize, D])  # [B*(N//blocksize), blocksize, D]
            descs2 = descs2.reshape([B * n_chunks, self.blocksize, D])  # [B*(N//blocksize), blocksize, D]
            valid_matches = valid_matches.view([B * n_chunks, self.blocksize])
            loss = self.criterion(descs1, descs2, valid_matches, euc=euc)
            if self.withconf:
                confs1, confs2 = map(lambda x: x[matches_perm], (confs1, confs2))  # apply perm to confidences if needed

        if self.withconf:
            # split confidences between positives/negatives for loss computation
            details['conf_pos'] = map(lambda x: x[valid_matches.view(B, -1)], (confs1, confs2))
            details['conf_neg'] = map(lambda x: x[~valid_matches.view(B, -1)], (confs1, confs2))
            details['Conf1_std'] = confs1[valid_matches.view(B, -1)].std()
            details['Conf2_std'] = confs2[valid_matches.view(B, -1)].std()

        return loss, details

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # Gather preds and GT
        descs1, descs2, confs1, confs2, valid_matches, monitoring = self.get_matching_descs(
            gt1, gt2, pred1, pred2, **kw)

        # loss on matches
        loss, details = self.blockwise_criterion(descs1, descs2, confs1, confs2,
                                                 valid_matches, euc=monitoring.pop('use_euclidean_dist', False))

        details[type(self).__name__] = float(loss.mean())
        return loss, (details | monitoring)
    
class NormalMetric(MultiLoss):
    def __init__(self):
        super().__init__()
        
    def get_name(self):
        return f'NormalMetric'
    
    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        assert "normal" in pred1 and "normal" in pred2
        gt_normal1 = gt1['normalmap']
        pred_normal1 = pred1['normal']
        gt_normal2 = gt2['normalmap']
        pred_normal2 = pred2['normal']
        
        mask1 = gt1['valid_mask'].clone()
        mask2 = gt2['valid_mask'].clone()
        
        dot1 = torch.cosine_similarity(pred_normal1, gt_normal1, dim=-1) # B,H,W
        valid_normal_mask1 = mask1.float() * (dot1.detach() < 0.999999).float() * (dot1.detach() > -0.999999).float()
        valid_normal_mask1 = valid_normal_mask1 > 0.0
        dot1 = dot1[valid_normal_mask1].clip(-1, 1)
        rmse1 = (((pred_normal1 - gt_normal1)**2).sum(axis=-1))[valid_normal_mask1]
        rmse1 = torch.sqrt(rmse1).mean()
        angle_err1 = (torch.acos(dot1) / torch.pi * 180).mean()
        
        dot2 = torch.cosine_similarity(pred_normal2, gt_normal2, dim=-1) # B,H,W
        valid_normal_mask2 = mask2.float() * (dot2.detach() < 0.999999).float() * (dot2.detach() > -0.999999).float()
        valid_normal_mask2 = valid_normal_mask2 > 0.0
        dot2 = dot2[valid_normal_mask2].clip(-1, 1)
        rmse2 = (((pred_normal2 - gt_normal2)**2).sum(axis=-1))[valid_normal_mask2]
        rmse2 = torch.sqrt(rmse2).mean()
        angle_err2 = (torch.acos(dot2) / torch.pi * 180).mean()
        
        details = {"rmse1": float(rmse1), 
                   "rmse2": float(rmse2),
                   "angle_err1": float(angle_err1), 
                   "angle_err2": float(angle_err2),
                   "rmse_count": float(rmse1 + rmse2)}
        
        return rmse1 + rmse2, details
        

class KappaNormalLoss(MultiLoss):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return f'KappaNormalLoss'
    
    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        assert "normal" in pred1 and "normal" in pred2
        
        gt_normal1 = gt1['normalmap']
        pred_normal1 = pred1['normal']
        kappa1 = pred1['kappa'] # B,H,W
        normal_mask1 = gt_normal1.abs().sum(-1) ## remove invalid normal
        
        gt_normal2 = gt2['normalmap']
        pred_normal2 = pred2['normal']
        kappa2 = pred2['kappa']
        normal_mask2 = gt_normal2.abs().sum(-1)
        
        mask1 = gt1['valid_mask'].clone()
        mask2 = gt2['valid_mask'].clone()
        
        # mask1, mask2: B,H,W
        dot1 = torch.cosine_similarity(pred_normal1, gt_normal1, dim=-1) # B,H,W
        valid_normal_mask1 = mask1.float() * (dot1.detach() < 0.999).float() * (dot1.detach() > -0.999).float()
        valid_normal_mask1 = valid_normal_mask1 * normal_mask1 > 0.0
        dot1 = dot1[valid_normal_mask1]
        kappa1 = kappa1[valid_normal_mask1]
        
        if dot1.shape[0] > 0:
            normal_loss1 = - torch.log(torch.square(kappa1) + 1) \
                            + kappa1 * torch.acos(dot1) \
                            + torch.log(1 + torch.exp(-kappa1 * np.pi))
            normal_loss1 = torch.mean(normal_loss1)
        else:
            normal_loss1 = torch.tensor(0.0).to(kappa1)
        
        dot2 = torch.cosine_similarity(pred_normal2, gt_normal2, dim=-1) # B,H,W
        valid_normal_mask2 = mask2.float() * (dot2.detach() < 0.999).float() * (dot2.detach() > -0.999).float()
        valid_normal_mask2 = valid_normal_mask2 * normal_mask2 > 0.0
        dot2 = dot2[valid_normal_mask2]
        kappa2 = kappa2[valid_normal_mask2]
        
        if dot2.shape[0] > 0:
            normal_loss2 = - torch.log(torch.square(kappa2) + 1) \
                        + kappa2 * torch.acos(dot2) \
                        + torch.log(1 + torch.exp(-kappa2 * np.pi))
            normal_loss2 = torch.mean(normal_loss2)
        else:
            normal_loss2 = torch.tensor(0.0).to(kappa2)

        details = {"normal_loss1": float(normal_loss1), 
                   "normal_loss2": float(normal_loss2),
                   "normal_loss_count": float(normal_loss1 + normal_loss2)}
        
        return normal_loss1 + normal_loss2, details

class ABSNormalLoss(MultiLoss):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return f'ABSNormalLoss'
    
    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        assert "normal" in pred1 and "normal" in pred2
        
        gt_normal1 = gt1['normalmap']
        pred_normal1 = pred1['normal']
        normal_mask1 = gt_normal1.abs().sum(-1) ## remove invalid normal
        
        gt_normal2 = gt2['normalmap']
        pred_normal2 = pred2['normal']
        normal_mask2 = gt_normal2.abs().sum(-1)
        
        mask1 = gt1['valid_mask'].clone()
        mask2 = gt2['valid_mask'].clone()
        valid_normal_mask1 = mask1 * normal_mask1 > 0.0
        valid_normal_mask2 = mask2 * normal_mask2 > 0.0
        
        if valid_normal_mask1.sum() > 0:
            loss_abs_1 = torch.abs(pred_normal1 - gt_normal1).sum(axis=-1)
            loss_abs_1 = loss_abs_1[valid_normal_mask1].mean()
        else:
            loss_abs_1 = torch.tensor(0.0).to(pred_normal1)
        
        if valid_normal_mask2.sum() > 0:
            loss_abs_2 = torch.abs(pred_normal2 - gt_normal2).sum(axis=-1)
            loss_abs_2 = loss_abs_2[valid_normal_mask2].mean()
        else:
            loss_abs_2 = torch.tensor(0.0).to(pred_normal2)
        
        details = {"normal_loss1": float(loss_abs_1), 
                   "normal_loss2": float(loss_abs_2),
                   "normal_loss_count": float(loss_abs_1 + loss_abs_2)}
        
        return loss_abs_1 + loss_abs_2, details
    
class SSIMNormalLoss(MultiLoss):
    def __init__(self):
        super().__init__()
        self.ssim_loss = SSIM(window_size=11, size_average=True)

    def get_name(self):
        return f'SSIMNormalLoss'
    
    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        assert "normal" in pred1 and "normal" in pred2
        
        gt_normal1 = gt1['normalmap']
        pred_normal1 = pred1['normal']
        normal_mask1 = gt_normal1.abs().sum(-1) ## remove invalid normal
        
        gt_normal2 = gt2['normalmap']
        pred_normal2 = pred2['normal']
        normal_mask2 = gt_normal2.abs().sum(-1)
        
        mask1 = gt1['valid_mask'].clone()
        mask2 = gt2['valid_mask'].clone()
        valid_normal_mask1 = mask1 * normal_mask1 > 0.0
        valid_normal_mask2 = mask2 * normal_mask2 > 0.0
        
        # set invalid normal zone to 0
        pred_normal1_copy = pred_normal1.clone()
        pred_normal2_copy = pred_normal2.clone()
        gt_normal1_copy = gt_normal1.clone()
        gt_normal2_copy = gt_normal2.clone()
        
        pred_normal1_copy[~valid_normal_mask1] = 0.0
        pred_normal2_copy[~valid_normal_mask2] = 0.0
        gt_normal1_copy[~valid_normal_mask1] = 0.0
        gt_normal2_copy[~valid_normal_mask2] = 0.0
        
        if valid_normal_mask1.sum() > 0:
            loss_ssim_1 = 1 - self.ssim_loss(pred_normal1_copy, gt_normal1_copy)
        else:
            loss_ssim_1 = torch.tensor(0.0).to(pred_normal1_copy)
        
        if valid_normal_mask2.sum() > 0:
            loss_ssim_2 = 1 - self.ssim_loss(pred_normal2_copy, gt_normal2_copy)
        else:
            loss_ssim_2 = torch.tensor(0.0).to(pred_normal2_copy)
        
        details = {"ssim_normal_loss1": float(loss_ssim_1), 
                   "ssim_normal_loss2": float(loss_ssim_2),
                   "ssim_normal_loss_count": float(loss_ssim_1 + loss_ssim_2)}
        
        return loss_ssim_1 + loss_ssim_2, details
    
class ABSNormalFromDepthLoss(MultiLoss):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return f'ABSNormalFromDepthLoss'
    
    def pcd2normal_torch(self, xyz):
        B, hd, wd, _ = xyz.shape 
        bottom_point = xyz[..., 2:hd,   1:wd-1, :]
        top_point    = xyz[..., 0:hd-2, 1:wd-1, :]
        right_point  = xyz[..., 1:hd-1, 2:wd,   :]
        left_point   = xyz[..., 1:hd-1, 0:wd-2, :]
        left_to_right = right_point - left_point
        bottom_to_top = top_point - bottom_point 
        # xyz_normal = np.cross(left_to_right, bottom_to_top, axis=-1)
        xyz_normal = torch.linalg.cross(left_to_right, bottom_to_top, dim=-1)
        
        # lefthand <<==>> righthand
        xyz_normal[..., 1] = xyz_normal[..., 1] * -1
        
        # norm = np.linalg.norm(xyz_normal, axis=-1, keepdims=True) + 1e-12
        # xyz_normal = xyz_normal / norm
        xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2.0, dim=-1, eps=1e-12, out=None)
        
        # xyz_normal = np.pad(xyz_normal, ((1,1),(1,1),(0,0)), mode='constant')
        xyz_normal = torch.nn.functional.pad(xyz_normal, (0, 0, 1, 1, 1, 1, 0, 0), mode='constant')
        return xyz_normal
    
    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        assert "pts3d" in pred1 and "pts3d_in_other_view" in pred2
        
        pts3d_1, pts3d_2 = pred1['pts3d'], pred2['pts3d_in_other_view'] # B,H,W,3
        
        gt_normal1 = gt1['normalmap']
        pred_normal1 = self.pcd2normal_torch(pts3d_1)
        normal_mask1 = gt_normal1.abs().sum(-1) ## remove invalid normal
        
        # gt_normal2 = gt2['normalmap']
        gt_normal2 = gt2['normalmap_transformed']
        pred_normal2 = self.pcd2normal_torch(pts3d_2)
        normal_mask2 = gt_normal2.abs().sum(-1)
        
        mask1 = gt1['valid_mask'].clone()
        mask2 = gt2['valid_mask'].clone()
        valid_normal_mask1 = (mask1 * normal_mask1) > 0.0
        valid_normal_mask2 = (mask2 * normal_mask2) > 0.0
        
        if valid_normal_mask1.sum() > 0:
            loss_abs_1 = torch.abs(pred_normal1 - gt_normal1).sum(axis=-1)
            loss_abs_1 = loss_abs_1[valid_normal_mask1].mean()
        else:
            loss_abs_1 = torch.tensor(0.0).to(pred_normal1)
        
        if valid_normal_mask2.sum() > 0:
            loss_abs_2 = torch.abs(pred_normal2 - gt_normal2).sum(axis=-1)
            loss_abs_2 = loss_abs_2[valid_normal_mask2].mean()
        else:
            loss_abs_2 = torch.tensor(0.0).to(pred_normal2)
        
        details = {"depth_normal_loss1": float(loss_abs_1), 
                   "depth_normal_loss2": float(loss_abs_2),
                   "depth_normal_loss_count": float(loss_abs_1 + loss_abs_2)}
        
        return loss_abs_1 + loss_abs_2, details
        

class ConfMatchingLoss(ConfLoss):
    """ Weight matching by learned confidence. Same as ConfLoss but for a matching criterion
        Assuming the input matching_loss is a match-level loss.
    """

    def __init__(self, pixel_loss, alpha=1., confmode='prod', neg_conf_loss_quantile=False):
        super().__init__(pixel_loss, alpha)
        self.pixel_loss.withconf = True
        self.confmode = confmode
        self.neg_conf_loss_quantile = neg_conf_loss_quantile

    def aggregate_confs(self, confs1, confs2):  # get the confidences resulting from the two view predictions
        if self.confmode == 'prod':
            confs = confs1 * confs2 if confs1 is not None and confs2 is not None else 1.
        elif self.confmode == 'mean':
            confs = .5 * (confs1 + confs2) if confs1 is not None and confs2 is not None else 1.
        else:
            raise ValueError(f"Unknown conf mode {self.confmode}")
        return confs

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # compute per-pixel loss
        loss, details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)
        # Recover confidences for positive and negative samples
        conf1_pos, conf2_pos = details.pop('conf_pos')
        conf1_neg, conf2_neg = details.pop('conf_neg')
        conf_pos = self.aggregate_confs(conf1_pos, conf2_pos)

        # weight Matching loss by confidence on positives
        conf_pos, log_conf_pos = self.get_conf_log(conf_pos)
        conf_loss = loss * conf_pos - self.alpha * log_conf_pos
        # average + nan protection (in case of no valid pixels at all)
        conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0
        # Add negative confs loss to give some supervision signal to confidences for pixels that are not matched in GT
        if self.neg_conf_loss_quantile:
            conf_neg = torch.cat([conf1_neg, conf2_neg])
            conf_neg, log_conf_neg = self.get_conf_log(conf_neg)

            # recover quantile that will be used for negatives loss value assignment
            neg_loss_value = torch.quantile(loss, self.neg_conf_loss_quantile).detach()
            neg_loss = neg_loss_value * conf_neg - self.alpha * log_conf_neg

            neg_loss = neg_loss.mean() if neg_loss.numel() > 0 else 0
            conf_loss = conf_loss + neg_loss

        return conf_loss, dict(matching_conf_loss=float(conf_loss), **details)
