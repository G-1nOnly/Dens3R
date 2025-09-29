# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------

# Modifications Copyright (C) <Alibaba Group>
# Changes: add tensorboard log in train process
# This is an adaptation and is distributed under the same license (CC BY-NC-SA 4.0).
# SPDX-License-Identifier: CC-BY-NC-SA-4.0(non-commercial use only)
import tqdm
import torch
import copy
import numpy as np

from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.model import AsymmetricCroCo3DStereo, inf  # noqa: F401, needed when loading the model
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf, inv
from matplotlib import pyplot as pl
from dust3r.utils.image import normal2color, pcd2normal_numpy, get_colormap
from dust3r.utils.image import colorize, inverse_normalize
cmap = pl.get_cmap('jet')
seg_cmap = get_colormap(40+1)


def load_model(model_path, device):
    print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    print(f"instantiating : {args}")
    net = eval(args) 
    print(net.load_state_dict(ckpt['model'], strict=False))
    return net.to(device)


def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2


def loss_of_one_batch(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None, 
                      bad_id_log_path=None, log_writer=None, global_step=None, log_img_step=300):
    view1, view2 = batch
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 
                       'true_shape', 'rng', 'model_id', 'size_info', 'size_info_matching', "rotate"])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        view1, view2 = make_batch_symmetric(batch)

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        pred1, pred2 = model(view1, view2)

        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(view1, view2, pred1, pred2) if criterion is not None else None
            
            if (log_writer is not None) and (global_step % log_img_step == 0):
                img1 = inverse_normalize(copy.deepcopy(view1["img"]))[0, ...]
                img2 = inverse_normalize(copy.deepcopy(view2["img"]))[0, ...]
                img_concat = torch.cat((img1, img2), dim=-1)
                log_writer.add_image(f"image_debug", img_concat, global_step=global_step, dataformats='CHW')
                
                if 'depth' in pred1 and 'depth' in pred2:
                    raw_depth_pred1 = pred1['depth'].clone()[0, ...].squeeze().detach().cpu().numpy()
                    raw_depth_pred2 = pred2['depth'].clone()[0, ...].squeeze().detach().cpu().numpy()
                    
                    raw_depth_pred1_color = colorize(raw_depth_pred1, vmin=0, vmax=raw_depth_pred1.max(), cmap='Spectral')[..., :3]
                    raw_depth_pred2_color = colorize(raw_depth_pred2, vmin=0, vmax=raw_depth_pred2.max(), cmap='Spectral')[..., :3]
                    pred_depth_concat = np.hstack((raw_depth_pred1_color, raw_depth_pred2_color))
                    
                    in_camera1 = inv(view1['camera_pose'])
                    gt_pts1 = geotrf(in_camera1, view1['pts3d'])  # B,H,W,3
                    gt_depth1 = gt_pts1[0, :, :, -1].detach().cpu().numpy()
                    in_camera2 = inv(view2['camera_pose'])
                    gt_pts2 = geotrf(in_camera2, view2['pts3d'])  # B,H,W,3
                    gt_depth2 = gt_pts2[0, :, :, -1].detach().cpu().numpy()
                    
                    gt_depth1_color = colorize(gt_depth1, vmin=0, vmax=gt_depth1.max(), cmap='Spectral')[..., :3]
                    gt_depth2_color = colorize(gt_depth2, vmin=0, vmax=gt_depth2.max(), cmap='Spectral')[..., :3]
                    gt_depth_concat = np.hstack([gt_depth1_color, gt_depth2_color])
                    
                    depth_concat = np.vstack([pred_depth_concat, gt_depth_concat])
                    log_writer.add_image(f"depth_pred", depth_concat, global_step=global_step, dataformats='HWC')

                if 'normal' in pred1 and 'normal' in pred2:
                    pred1_normal = pred1['normal'].clone()[0, ...].detach().cpu().numpy()
                    pred2_normal = pred2['normal'].clone()[0, ...].detach().cpu().numpy()
                    normal1_color = normal2color(pred1_normal)
                    normal2_color = normal2color(pred2_normal)
                    normal_concat = np.hstack((normal1_color, normal2_color))
                else:
                    normal_concat = None
                
                if 'pts3d' in pred1:
                    # visualize normal from depth
                    points_img1 = (pred1['pts3d']).clone()[0, ...].detach().cpu().numpy()
                    points_img2 = (pred2['pts3d_in_other_view']).clone()[0, ...].detach().cpu().numpy()
                    normal1_from_depth = pcd2normal_numpy(points_img1)
                    normal1_from_depth_color = normal2color(normal1_from_depth)
                    normal2_from_depth = pcd2normal_numpy(points_img2)
                    normal2_from_depth_color = normal2color(normal2_from_depth)
                    normal_from_color_concat = np.hstack((normal1_from_depth_color, normal2_from_depth_color))
                else:
                    normal_from_color_concat = None
                
                gt1_normal = view1["normalmap"].clone()[0, ...].detach().cpu().numpy()
                if "normalmap_transformed" in view2:
                    gt2_normal = view2["normalmap_transformed"].clone()[0, ...].detach().cpu().numpy()
                else:
                    gt2_normal = view2["normalmap"].clone()[0, ...].detach().cpu().numpy()
                normal1_color_gt = normal2color(gt1_normal)
                normal2_color_gt = normal2color(gt2_normal)
                normal_all_concat = np.hstack((normal1_color_gt, normal2_color_gt))
                
                if normal_concat is not None:
                    normal_all_concat = np.concatenate((normal_all_concat, normal_concat), axis=0)
                if normal_from_color_concat is not None:
                    normal_all_concat = np.concatenate((normal_all_concat, normal_from_color_concat), axis=0)
                
                log_writer.add_image(f"normal_debug", normal_all_concat, global_step=global_step, dataformats='HWC')

    result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=loss)
    return result[ret] if ret else result


@torch.no_grad()
def inference(pairs, model, device, batch_size=8, verbose=True):
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, None, device)
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    return result


def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


def get_pred_pts3d(gt, pred, use_pose=False):
    if 'depth' in pred and 'pseudo_focal' in pred:
        try:
            pp = gt['camera_intrinsics'][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif 'pts3d' in pred:
        # pts3d from my camera
        pts3d = pred['pts3d']

    elif 'pts3d_in_other_view' in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred['pts3d_in_other_view']  # return!

    if use_pose:
        camera_pose = pred.get('camera_pose')
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(gt_pts1, gt_pts2, pr_pts1, pr_pts2=None, fit_mode='weiszfeld_stop_grad', valid1=None, valid2=None):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    # concat the pointcloud
    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None

    all_gt = torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1) if gt_pts2 is not None else nan_gt_pts1
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith('avg'):
        # scaling = (all_pr / all_gt).view(B, -1).mean(dim=1)
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith('median'):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith('weiszfeld'):
        # init scaling with l2 closed form
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip_(min=1e-8).reciprocal()
            # update the scaling with the new weights
            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f'bad {fit_mode=}')

    if fit_mode.endswith('stop_grad'):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)
    # assert scaling.isfinite().all(), bb()
    return scaling
