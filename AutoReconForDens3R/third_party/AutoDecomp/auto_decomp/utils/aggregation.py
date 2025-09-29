from collections import defaultdict
from typing import Dict

import numpy as np
import pycolmap
import torch
from torch import Tensor
from torch.nn.functional import grid_sample

from auto_decomp.utils import tqdm as tqdm_utils


def multiview_feature_extraction(
    rec,
    mv_feats: Dict[str, Dict[int, Tensor]],
    reproject_3d_points: bool = True,
    interpolate="nearest",
    align_corners=None,
    max_reproj_error=6.0,
    min_track_length=2,
    device="cpu",
):
    """Extract multi-view features with the reconstructed SfM point cloud.
    Args:
        mv_feats (Dict[str, Dict[int, Tensor]]): {feature_name: {image_id: (C, H, W)}} multi-view features
    Returns:
        p3Ds_xyz (Tensor): (L, 3)
        p3D_mv_feats (Dict[str, List[Tensor]]): {feature_name: [L, (M, C)]}
    """
    if not isinstance(rec, pycolmap.Reconstruction):
        rec = pycolmap.Reconstruction(rec)

    # filter outliers
    bbs = rec.compute_bounding_box(0.001, 0.999)
    p3Ds_filtered = {
        p3D_id: p3D
        for p3D_id, p3D in rec.points3D.items()
        if (
            (p3D.xyz >= bbs[0]).all()
            and (p3D.xyz <= bbs[1]).all()
            and p3D.error <= max_reproj_error
            and p3D.track.length() >= min_track_length
        )
    }

    # parse registered images
    image_ids = rec.reg_image_ids()  # regsitered images

    # interpolate mv_feats for each p3D
    # (feat, reproj_error)
    p3D_id_to_feats = {k: defaultdict(list) for k in mv_feats.keys()}  # {feature_name: {p3D_id: List[Tensor]}}
    for img_id in tqdm_utils.tqdm(image_ids, desc="Aggregating multi-view colors/features"):
        img = rec.images[img_id]
        cam = rec.cameras[img.camera_id]
        img_h, img_w = cam.height, cam.width
        p3D_ids = [p2D.point3D_id for p2D in img.points2D if p2D.has_point3D()]
        if reproject_3d_points:
            p3Ds = [rec.points3D[p3D_id] for p3D_id in p3D_ids]
            xyzs = np.array([p3D.xyz for p3D in p3Ds])
            p2Ds = torch.tensor(
                np.array(cam.world_to_image(img.project(xyzs))), dtype=torch.float32, device=device
            )  # (N, 2), <x, y>, cpix
        else:
            p2Ds = torch.tensor(
                [p2D.xy for p2D in img.points2D if p2D.has_point3D()], dtype=torch.float32, device=device
            )  # cpix

        if len(p2Ds) == 0:
            continue

        p2Ds_normalized = (p2Ds / torch.tensor([[img_w, img_h]], dtype=torch.float32, device=device) - 0.5) * 2
        p2Ds_normalized = p2Ds_normalized[None, None]  # (1, 1, N, 2)

        # interpolate feature map
        # TODO: call grid_sample() only once for all features
        for feat_name, _p3D_id_to_feats in p3D_id_to_feats.items():
            feats = grid_sample(
                mv_feats[feat_name][img_id][None], p2Ds_normalized, mode=interpolate, align_corners=align_corners
            )  # (1, C, 1, N)
            feats = feats[0, :, 0].transpose(0, 1)  # (N, C)
            for p3D_id, feat in zip(p3D_ids, feats):
                _p3D_id_to_feats[p3D_id].append(feat)  # type: ignore
    p3D_id_to_feats = {
        feat_name: {k: torch.stack(v, 0) for k, v in _p3D_id_to_feats.items()}
        for feat_name, _p3D_id_to_feats in p3D_id_to_feats.items()
    }

    p3Ds_filtered_xyz, p3Ds_filtered_feats = [], defaultdict(list)
    for p3D_id, p3D in sorted(p3Ds_filtered.items(), key=lambda x: x[0]):
        p3Ds_filtered_xyz.append(p3D.xyz)
        for feat_name, _p3D_id_to_feats in p3D_id_to_feats.items():
            p3Ds_filtered_feats[feat_name].append(_p3D_id_to_feats[p3D_id])  # type: ignore
    p3Ds_filtered_xyz = torch.tensor(np.array(p3Ds_filtered_xyz), device=device)
    return p3Ds_filtered_xyz, p3Ds_filtered_feats


def multiview_feature_aggregation(p3Ds_xyz, p3Ds_mv_feats, mode="mean", affinity_agg_mode="max", gini_agg_mode="max"):
    """
    Args:
        p3Ds_xyz (Tensor): (L, 3)
        p3Ds_mv_feats (Dict[str, List[Tensor])]: {feature_name: [L, (M, C)]}
    Returns:
        p3Ds_feat (Dict[str, Tensor]): {feature_name: (L, C)}
    """

    def _aggregate_mv_feats(p3Ds_xyzm, _p3Ds_mv_feats):
        _p3Ds_feat = []
        for mv_feats in _p3Ds_mv_feats:
            if mode == "mean":
                feat = mv_feats.mean(dim=0)  # (C, )
            elif mode == "max":
                feat = mv_feats.max(dim=0).values  # (C, )
            elif mode == "gini":
                feat = gini_feature_aggregation(mv_feats, gini_agg_mode)
            elif mode == "affinity":
                feat = affinity_feature_aggregation(mv_feats, affinity_agg_mode)
            else:
                raise ValueError(f"Unknown aggregation mode: {mode}")
            _p3Ds_feat.append(feat)
        _p3Ds_feat = torch.stack(_p3Ds_feat, dim=0)  # (L, C)
        return _p3Ds_feat

    p3Ds_feat = {
        feat_name: _aggregate_mv_feats(p3Ds_xyz, _p3Ds_mv_feats) for feat_name, _p3Ds_mv_feats in p3Ds_mv_feats.items()
    }

    return p3Ds_feat


def gini(array, eps=1e-7):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += eps
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def gini_feature_aggregation(mv_feats, mode="max"):
    # assume salient feature has larger gini coefficients (higher sparsity)
    _mv_feats = mv_feats.clone().numpy()
    gini_coeffs = mv_feats.new_tensor([gini(f) for f in _mv_feats])
    if mode == "max":
        feat = mv_feats[torch.argmax(gini_coeffs)]
    elif mode == "weighted_mean":
        raise NotImplementedError()
    else:
        raise ValueError()
    return feat


def affinity_feature_aggregation(mv_feats, mode):
    raise NotImplementedError()
