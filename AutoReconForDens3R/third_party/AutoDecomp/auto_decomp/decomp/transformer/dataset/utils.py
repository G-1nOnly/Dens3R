from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from einops import repeat
from torch import Tensor
from torch.utils.data import default_collate

from auto_decomp.utils.geometry.box3d import convert_box_format
from auto_decomp.utils.geometry.pointcloud import NeuralPointCloud

BBOX_KEYS = ["bbox", "extended_bbox", "box_pseudo_gt_pca", "box_pseudo_gt_aabb"]


def normalize_pointcloud(data):
    # normalize pointcloud to a [-1, 1]^3 bbox (PointNet normalize ptcd into a unit-sphere)
    xyz = data["xyz"]
    _max, _min = xyz.max(0), xyz.min(0)
    scale = 1 / ((_max - _min).max() + 1e-7)
    xyz = (xyz - _min) * scale * 2 - 1
    assert xyz.min() >= -1 and xyz.max() <= 1
    data["xyz"] = xyz
    if "avg_cam_loc" in data:  # for testing only, no other augmentation allowed
        data["avg_cam_loc"] = (data["avg_cam_loc"] - _min) * scale * 2 - 1
    data["normalize_min"] = _min
    data["normalize_scale"] = scale  # for de-noramlization

    for k in ["bbox", "extended_bbox"]:
        if k in data:
            # FIXME: normalize_pointcloud is called twice and data[k].shape == (8, 3) for the 2nd call.
            # assert data[k].shape == (2, 3)  # (center, extent)
            if data[k].shape == (2, 3):
                data[k] = convert_box_format(data[k], "center_extent", "vertices")

    for k in BBOX_KEYS:
        if k in data:
            data[k] = (data[k] - _min) * scale * 2 - 1
            # if k != 'extended_bbox':
            #     assert data[k].min() >= -1 and data[k].max() <= 1, f'{k} not in [-1, 1]\n: {data[k]}'
    return data


def to_numpy(obj):
    if isinstance(obj, Tensor):
        obj = obj.cpu().numpy()
        if obj.dtype == np.float64:
            obj = obj.astype(np.float32)
        return obj
    elif isinstance(obj, np.ndarray):
        if obj.dtype == np.float64:
            obj = obj.astype(np.float32)
        return obj
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = to_numpy(v)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(to_numpy(v))
        return res
    else:
        return obj


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, str):
        return obj
    else:
        raise TypeError(f"Invalid type for move_to: {type(obj)}")


def build_inference_batch(
    data: Dict[str, Any], facet: str = "key", device: str = "cuda", normalize_ptcd: bool = True
) -> Dict[str, torch.Tensor]:
    """Build a batch for transformer inference.

    Args:
        data (Dict[str, Any]): the preprocessed sfm data
            {
                "inst_name": str
                "poses": Dict: {image_name: {'T44_w2c': np.ndarray, 'K33': np.ndarray}}
                "neural_ptcd": NeuralPointCloud
            }
    """
    batch = {}

    inst_name, inst_dir, poses, neural_ptcd = data["inst_name"], data["inst_dir"], data["poses"], data["neural_ptcd"]
    xyz = neural_ptcd.p3Ds  # (n_p, 3)
    feat = neural_ptcd.pointwise_features[NeuralPointCloud.MAIN_FEATURE_KEY]  # (n_p, nc)
    cls_feat = neural_ptcd.imagewise_features[f"cls_{facet}"]  # (nc,)

    avg_cam_loc = compute_average_camera_position(poses)

    batch = {
        "xyz": xyz,
        "feat": feat,
        "cls_feat": cls_feat,
        "poses": poses,
        "avg_cam_loc": avg_cam_loc,
        "inst_name": inst_name,
        "inst_dir": str(inst_dir)
    }
    if normalize_ptcd:
        batch = normalize_pointcloud(to_numpy(batch))
    batch = default_collate([batch])
    batch = move_to(batch, device)
    return batch


def compute_average_camera_position(poses: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    all_T44_w2c = np.stack([v["T44_w2c"] for v in poses.values()], 0)
    all_T44_c2w = repeat(np.eye(4), "a b -> c a b", c=len(all_T44_w2c))
    all_T44_c2w[:, :3, :3] = all_T44_w2c[:, :3, :3].transpose(0, 2, 1)
    all_T44_c2w[:, :3, [-1]] = -all_T44_c2w[:, :3, :3] @ all_T44_w2c[:, :3, [-1]]
    avg_cam_loc = all_T44_c2w[:, :3, -1].mean(0)  # (3, )
    return avg_cam_loc
