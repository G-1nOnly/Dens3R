# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data parser for datasets in AutoRecon format."""
from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import cv2
import numpy as np
import torch
import open3d as o3d
from einops import rearrange
from PIL import Image
from rich.console import Console
from scipy.spatial import ConvexHull
from skimage.draw import polygon2mask
from sklearn.decomposition import PCA
from torch.nn import functional as F
from torch_cluster import fps as fps_torch
from torchtyping import TensorType
from tqdm import tqdm
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    Semantics,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.pointclouds import BasicPointClouds

try:
    from pytorch3d.ops.knn import knn_points
except ImportError as ie:
    print(ie)


CONSOLE = Console()


def glob_data(pattern: str) -> List[str]:
    data_paths = []
    data_paths.extend(glob(pattern))
    data_paths = sorted(data_paths)
    return data_paths


def get_src_from_pairs(
    ref_idx, all_imgs, pairs_srcs, neighbors_num=None, neighbors_shuffle=False
) -> Dict[str, TensorType]:
    # src_idx[0] is ref img
    src_idx = pairs_srcs[ref_idx]
    # randomly sample neighbors
    if neighbors_num and neighbors_num > -1 and neighbors_num < len(src_idx) - 1:
        if neighbors_shuffle:
            perm_idx = torch.randperm(len(src_idx) - 1) + 1
            src_idx = torch.cat([src_idx[[0]], src_idx[perm_idx[:neighbors_num]]])
        else:
            src_idx = src_idx[: neighbors_num + 1]
    src_idx = src_idx.to(all_imgs.device)
    return {"src_imgs": all_imgs[src_idx], "src_idxs": src_idx}


def get_image(image_filename, alpha_color=None) -> TensorType["image_height", "image_width", "num_channels"]:
    """Returns a 3 channel image.

    Args:
        image_idx: The image index in the dataset.
    """
    pil_image = Image.open(image_filename)
    np_image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
    assert len(np_image.shape) == 3
    assert np_image.dtype == np.uint8
    assert np_image.shape[2] in [3, 4], f"Image shape of {np_image.shape} is in correct."
    image = torch.from_numpy(np_image.astype("float32") / 255.0)
    if alpha_color is not None and image.shape[-1] == 4:
        assert image.shape[-1] == 4
        image = image[:, :, :3] * image[:, :, -1:] + alpha_color * (1.0 - image[:, :, -1:])
    else:
        image = image[:, :, :3]
    return image


def get_mask(image_idx, mask_paths):
    """read binary segmentation masks"""
    p = mask_paths[image_idx]
    mask = cv2.imread(str(p))
    if mask is None:
        raise FileNotFoundError(p)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (torch.from_numpy(mask) > 0).to(torch.long)[..., None]

    return {"binary_mask": mask}


def load_aabb_from_obj_scale_mat(obj_scale_mat: np.ndarray, scale_mat: Optional[np.ndarray] = None) -> np.ndarray:
    # scale_mat: normalized world -> original world
    # obj_scale_mat: unit_cube in normalized_world -> scale_cube in ori_world
    obj_bbox_min = np.array([-1.0, -1.0, -1.0, 1.0])
    obj_bbox_max = np.array([1.0, 1.0, 1.0, 1.0])
    if scale_mat is None:
        scale_mat = np.eye(4, dtype=np.float32)
    obj_bbox_min = np.linalg.inv(scale_mat) @ obj_scale_mat @ obj_bbox_min[:, None]
    obj_bbox_max = np.linalg.inv(scale_mat) @ obj_scale_mat @ obj_bbox_max[:, None]
    obj_aabb = torch.tensor(np.stack([obj_bbox_min[:3, 0], obj_bbox_max[:3, 0]], 0), dtype=torch.float32)
    return obj_aabb

def load_dens3r_normal(image_idx, normal_filenames, camera_to_worlds):
    """
    """
    normal_path = normal_filenames[image_idx]
    data = np.load(normal_path)
    normal = data[list(data.keys())[0]] # "arr_0"
    normal = torch.from_numpy(normal).float()

    rot = camera_to_worlds[image_idx][:3, :3].clone()
    
    # handle axis transform
    magic_rot = torch.tensor([[-1,0,0],[0,1,0],[0,0,-1]], dtype=torch.float32)
    
    # rot = rot @ torch.tensor([[1,0,0],[0,-1,0],[0,0,-1]], dtype=torch.float32)
    # magic_rot = torch.tensor([[1,0,0],[0,-1,0],[0,0,1]], dtype=torch.float32)
    
    # magic_rot = torch.tensor([[1,0,0],[0,1,0],[0,0,-1]], dtype=torch.float32)

    normal_map = torch.nn.functional.normalize(normal.reshape(-1, 3), p=2, dim=-1)
    normal_map = ((normal_map @ magic_rot.T) @ rot.T) 
    normal_map = normal_map.reshape(*normal.shape[:-1], 3)

    output = {"normal": normal_map}

    return output

def compute_extended_object_bounds(
    obj_bbox_len: float, extend_mode: Literal["all_sides", "except_bottom", "only_bottom", "only_top"]
):
    # NOTE: obj_bbox_len is the normalized length (homo cube), prior to applying scale_mats
    if extend_mode == "all_sides":
        obj_bbox_min = np.array([-obj_bbox_len, -obj_bbox_len, -obj_bbox_len, 1.0])
        obj_bbox_max = np.array([obj_bbox_len, obj_bbox_len, obj_bbox_len, 1.0])
    elif extend_mode == "only_bottom":  # assume +y is down
        obj_bbox_min = np.array([-1.0, -1.0, -1.0, 1.0])
        obj_bbox_max = np.array([1.0, obj_bbox_len, 1.0, 1.0])
    elif extend_mode == "only_up":
        obj_bbox_min = np.array([-1.0, -obj_bbox_len, -1.0, 1.0])
        obj_bbox_max = np.array([1.0, 1.0, 1.0, 1.0])
    elif extend_mode == "except_bottom":  # assume +y is down
        obj_bbox_min = np.array([-obj_bbox_len, -obj_bbox_len, -obj_bbox_len, 1.0])
        obj_bbox_max = np.array([obj_bbox_len, 1.0, obj_bbox_len, 1.0])
    else:
        raise ValueError(f"Unknown object_bounds_extend_mode: {extend_mode}")

    return obj_bbox_min, obj_bbox_max


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose  # T44_c2w


def project_3d_bbox_to_frames(
    aabb: TensorType[2, 3], all_K33: TensorType["n_frames", 3, 3], all_T44_c2w: TensorType["n_frames", 4, 4]
) -> TensorType["n_frames", 8, 2]:
    """project 3d aabb vertices to frames"""
    _dev = aabb.device

    bbox_pts = aabb.new_ones((8, 4))  # homo pts
    _indices = torch.tensor(
        [
            [[0, 0], [0, 1], [0, 2]],
            [[0, 0], [1, 1], [0, 2]],
            [[0, 0], [0, 1], [1, 2]],
            [[0, 0], [1, 1], [1, 2]],
            [[1, 0], [0, 1], [0, 2]],
            [[1, 0], [1, 1], [0, 2]],
            [[1, 0], [0, 1], [1, 2]],
            [[1, 0], [1, 1], [1, 2]],
        ],
        device=_dev,
        dtype=torch.long,
    )  # (8, 3, 2)
    _indices = _indices.reshape(-1, 2)  # (24, 2)
    bbox_pts[:, :3] = aabb[_indices[:, 0], _indices[:, 1]].reshape(8, 3)
    all_T44_w2c = all_T44_c2w.clone()
    all_R33_w2c = all_T44_c2w[:, :3, :3].permute(0, 2, 1)
    all_t31_w2c = -torch.bmm(all_R33_w2c, all_T44_c2w[:, :3, [-1]])
    all_T44_w2c[:, :3, :] = torch.cat([all_R33_w2c, all_t31_w2c], dim=-1)
    all_K44 = torch.eye(4, dtype=all_K33.dtype, device=_dev)[None].repeat(len(all_K33), 1, 1)
    all_K44[:, :3, :3] = all_K33
    all_KRT44 = torch.bmm(all_K44, all_T44_w2c)
    _bbox_pts = torch.einsum("krc,nc->knr", all_KRT44, bbox_pts)[..., :3]  # (n_frames, 8, 3)
    bbox_pts_2d = _bbox_pts[..., :2] / _bbox_pts[..., [-1]]  # (n_frames, 8, 2)
    return bbox_pts_2d  # (n_frames, 8, 2)


def compute_knn_distance_threshold(points, kdtree=None, n_neighbors=10, std_ratio=0.5):
    if kdtree is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    mean_dists = []
    for p in points:
        _, inds, _ = kdtree.search_knn_vector_3d(p, n_neighbors)
        mean_dist = np.linalg.norm(p - points[inds], axis=1, ord=2).mean(0)
        mean_dists.append(mean_dist)
    mean_dists = np.array(mean_dists)
    avg_mean_dist = mean_dists.mean()
    std = np.std(mean_dists)

    max_dist_threshold = avg_mean_dist + std * std_ratio
    return max_dist_threshold


def points2polygon(pts: TensorType["bs", "n_pts", 2]) -> List[np.ndarray]:
    """compute convex hull of the given points"""
    polygons = []  # convexhulls
    for _pts in pts.cpu().numpy():
        hull = ConvexHull(_pts)
        polygons.append(_pts[hull.vertices, :])
    return polygons  # [N, (K', 2)]


def compute_ptwise_knn_distance_threshold(pts: TensorType["n", 3], knn: int, std_ratio: float) -> TensorType["n"]:
    knn_rets = knn_points(pts[None], pts[None], K=knn, return_nn=True)
    # knn_pts = knn_gather(pts[None], knn_rets.idx)[0]  # (n, k, 3)
    _, knn_dists = knn_rets.idx[0], torch.sqrt(knn_rets.dists[0])  # (n, k)
    knn_dists_mean, knn_dists_std = knn_dists.mean(-1), knn_dists.std(-1)  # (n, )
    if knn == 1:
        knn_dists_std = torch.zeros_like(knn_dists_mean)
    knn_dists_thr = knn_dists_mean + knn_dists_std * std_ratio
    return knn_dists_thr


def get_depths_and_normals(image_idx: int, depths, normals):
    """function to process additional depths and normal information

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """

    # depth
    depth = depths[image_idx]
    # normal
    normal = normals[image_idx]

    return {"depth": depth, "normal": normal}


def get_features(image_idx: int, features: torch.Tensor, pca: PCA):
    """process image features"""
    feature = features[image_idx]  # (c, h, w)
    return {"feature": feature, "gt_features_pca": pca}


def get_features_pca(image_idx: int, features_pca: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor):
    """process pca results of image features"""
    feature_pca = features_pca[image_idx]  # (c, h, w)
    return {"feature_pca": feature_pca, "feature_pca_min": min_val, "feature_pca_max": max_val}


@dataclass
class AutoReconDataParserConfig(DataParserConfig):
    """AutoRecon dataset parser config"""

    _target: Type = field(default_factory=lambda: AutoRecon)
    """target class to instantiate"""
    data: Path = Path("data/auto_recon/DTU/dtu_scan65")
    """Directory specifying location of data."""
    train_split_percentage: Optional[float] = None
    """Percentage of data to use for training. Load all data for all splits if not specified."""
    parse_images_from_camera_dict: bool = False
    """Parse training images from camera npz, otherwise parse all images from image directory"""
    image_dirname: str = "image"
    image_extension: str = ".png"  # ".jpg" for CO3D & bmvs_volsdf
    anno_dirname: str = "sfm-transfer_sdfstudio-dff"
    """Name of directory containing camera_filename and object_filename"""
    camera_filename: str = "cameras_sfm-transfer_norm-obj-side-2.0.npz"
    """.npz file containing camera intrinsics & extrinsics"""
    object_filename: str = "objects_sfm-transfer_norm-obj-side-2.0.npz"
    """.npz file containing object sizes, fg & bg points and features, plane params, etc."""
    object_bounds_extend_ratio: float = 0.0
    """extend object aabb along each axis by this ratio (assume +y is down)"""
    object_bounds_extend_mode: Literal["all_sides", "except_bottom", "only_bottom"] = "all_sides"
    """the specific mode when extending object bounds"""
    # TODO: allow specifying the gound direction manually
    # extend_ground_plane_only: bool = False
    # """when extend a object bounds, only extend the bottom one"""
    include_mono_prior: bool = False
    """whether or not to include loading of normal """
    center_crop_type: Literal[
        "center_crop_for_replica", "center_crop_for_tnt", "center_crop_for_dtu", "no_crop"
    ] = "no_crop"
    """center crop type as monosdf, we should create a dataset that don't need this"""
    load_pairs: bool = False
    """whether to load pairs for multi-view consistency"""
    neighbors_num: Optional[int] = None
    neighbors_shuffle: Optional[bool] = False
    pairs_sorted_ascending: Optional[bool] = True
    """if src image pairs are sorted in ascending order by similarity i.e. the last element is the most similar to the first (ref)"""
    include_image_features: bool = True
    """whether to load image features for DFF training"""
    feature_dirname: str = "dino_feats_172800"
    feature_key: str = "features"
    include_coarse_pointclouds: bool = True
    """load pointclouds from corase decomposition"""
    include_coarse_features: bool = True
    """load pointcloud features from corase decomposition"""
    downsample_ptcd: int = 1000  # TODO: tune the number of donwsampled fg, bg and plane points
    """donwsample pointclouds to this number, -1 for no downsampling"""
    decomposition_mode: Literal["regularization", "segmentation"] = "segmentation"
    """which decomposition mode to load data for"""
    use_ori_res_image: bool = True
    """use images w/ resolutions of the original dataset (e.g., 1600x1200 for DTU)"""
    collider_type: Literal["box", "near_far", "box_near_far", "sphere"] = "box_near_far"
    """set the collider type for ray sampling"""
    near_far: Tuple[float, float] = (0.1, 6.0)
    """set the (near, far) used by ray collider.
    If collider_type == box_near_far / sphere, then this would set (near, far) for the background ray sampler;
    If collider_type == box, then far would be used as near for the background ray sampler.
    """
    use_accurate_scene_box: bool = True
    """If True, set the scene box to the aabb of the object aabb;
    otherwise, set the scene box to the unit-cube, which is consistent with the foreground region
    of SceneContraction('inf')
    """
    aabb_scale: float = 1.0
    """scale the size of the scene box with this factor, use a larger aabb_scale to include
    more background points into the scene box
    """
    scale_factor: float = 1.0
    """scale the camera origins and the scene box with this factor, this would affect contraction
    boundary of MipNeRF-360 SceneContractio.
    """
    scale_aabb_to_unit_cube: bool = False
    """if the object aabb is bigger than a unit cube, scale it to a unit cube, otherwise truncate the aabb"""
    scale_aabb_to_unit_sphere: bool = False
    """scale the scene box inside a unit sphere, this is helpful when using scene contraction w/ l2-norm"""
    include_fg_mask: bool = False
    """load additional foreground masks for training / segmentation evaluation."""
    compute_fg_bbox_mask: bool = False
    """compute per-frame masks of the foreground 3d bbox, ignored if masks already exists in fg_bbox_mask_dirname"""
    force_recompute_fg_bbox_mask: bool = False
    """force recompute per-frame masks of the foreground 3d bbox, even if masks already exists in fg_bbox_mask_dirname"""
    fg_bbox_mask_dirname: str = "fg_bbox_mask"
    """directory name for saving per-frame masks corresponding to the foreground 3d bbox"""

    # pointcloud regularization configs
    ptcd_reg_fg_global_knn: int = 10
    ptcd_reg_fg_global_std_ratio: float = 0.5
    """fg_sdf_deviation_thr = mean(knn distances) + std(knn distances) * std_ratio"""
    ptcd_reg_fg_ptwise_knn: int = 10
    ptcd_reg_fg_ptwise_std_ratio: float = 0.5
    ptcd_reg_plane_ptwise_knn: int = 10
    ptcd_reg_plane_ptwise_std_ratio: float = 0.5

    include_segmentation: bool = False
    """load segmentation """
    segmentation_dirname: str = "tokencut_saliency/bfs_masks"
    
    normal_dirname: str = "dens3r_normal"

    # TODO: maintain different scene boxes for reconstruction and foreground segmentation
    # (or set object_bounds_extend_ratio to negative values to shrink )

    def __post_init__(self):
        if not self.use_ori_res_image and self.parse_images_from_camera_dict:
            raise ValueError


@dataclass
class AutoRecon(DataParser):
    """AutoRecon dataset parser."""

    config: AutoReconDataParserConfig

    def __post_init__(self):
        self.data_root = self.config.data
        self.anno_dir = self.data_root / self.config.anno_dirname
        self.camera_anno_path = self.anno_dir / self.config.camera_filename
        self.object_anno_path = self.anno_dir / self.config.object_filename  # aabb, fg & bg points, features, etc.
        self.image_dir = self.data_root / self.config.image_dirname
        self.feat_dir = self.data_root / self.config.feature_dirname
        self.fg_bbox_mask_dir = self.data_root / self.config.fg_bbox_mask_dirname
        self.seg_dir = self.data_root / self.config.segmentation_dirname
        self.normal_dir = self.data_root / self.config.normal_dirname

        if self.config.scale_aabb_to_unit_sphere:
            assert self.config.scale_factor == 1.0
            self.scale_factor = None  # to be computed later
        else:
            self.scale_factor = self.config.scale_factor

    def _generate_dataparser_outputs(self, split="train"):  # pylint: disable=unused-argument,too-many-statements
        # load images
        image_paths = self._load_images()
        n_images = len(image_paths)
        if n_images == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

        scale_mats, all_T44_c2w, fx, fy, cx, cy = self._load_poses(n_images)
        image_paths, all_T44_c2w, intrinsics, scale_mats = self._parse_split(
            image_paths, all_T44_c2w, torch.stack([fx, fy, cx, cy], -1), scale_mats, split=split
        )
        fx, fy, cx, cy = torch.split(intrinsics, 1, dim=-1)

        # Build cameras & scnene box
        scale_mat_for_obj_anno = None
        scene_box = self._build_scene_box(scale_mat=scale_mat_for_obj_anno)
        cameras = self._build_cameras(image_paths, all_T44_c2w, fx, fy, cx, cy)
        mask_paths = self._compute_fg_bbox_masks(all_T44_c2w, cameras.get_intrinsics_matrices(), image_paths)

        # Load additional inputs
        additional_inputs_dict, metadata = {}, {}
        self._load_image_features(additional_inputs_dict, metadata, image_paths=image_paths)
        self._load_pointclouds(metadata, scale_mat=scale_mat_for_obj_anno)
        # TODO: raise NotImplementedError if train-val-split is specified
        # depth_images, normal_images = self._load_mono_prior(all_T44_c2w, additional_inputs_dict)
        normal_paths = [str(self.normal_dir / Path(img_path).stem) + "_normal.npz" for img_path in image_paths]
        depth_images, normal_images = self._load_dens3r_prior_without_load(all_T44_c2w, normal_paths, additional_inputs_dict)
        self._load_image_pairs(image_paths, additional_inputs_dict, split=split)
        self._load_segmentation(additional_inputs_dict, metadata, image_paths=image_paths)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_paths,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_paths,
            additional_inputs=additional_inputs_dict,
            metadata=metadata,
            depths=depth_images,
            normals=normal_images,
        )
        return dataparser_outputs

    def _parse_split(
        self,
        image_paths: List[Path],
        all_T44_c2w: np.ndarray,
        intrinsics: np.ndarray,
        scale_mats: np.ndarray,
        split: Literal["train", "test", "val"] = "train",
    ):
        """Parse train / val / test split."""
        if self.config.train_split_percentage is None or self.config.train_split_percentage == 1.0:
            return image_paths, all_T44_c2w, intrinsics, scale_mats

        num_images = len(image_paths)
        num_train_images = math.ceil(num_images * self.config.train_split_percentage)
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        if len(indices) == 0:
            raise RuntimeError(
                f"No image assigned for split {split} (train_split_percentage={self.config.train_split_percentage})"
            )
        CONSOLE.print(f"[green]Loading data for {split} split with {len(indices)}/{num_images} images")

        image_paths = [image_paths[i] for i in indices]
        all_T44_c2w = all_T44_c2w[indices]
        intrinsics = intrinsics[indices]
        scale_mats = scale_mats[indices]

        return image_paths, all_T44_c2w, intrinsics, scale_mats

    def _load_images(self):
        if self.config.parse_images_from_camera_dict:
            image_paths = self._parse_images_from_camera_dict()
        else:  # parse all images from image dir
            image_paths = self._parse_images_from_image_dir()
        self.n_images = len(image_paths)
        return image_paths

    def _parse_images_from_camera_dict(self):
        """Parse images from camera dict, which is useful when there are too many frames.

        TODO: images not in the camera dict can be used during validation.
        """
        camera_dict = np.load(self.camera_anno_path)
        suffix = next(self.image_dir.iterdir()).suffix
        _sorted_img_keys = sorted(
            [k for k in camera_dict.keys() if "image_name" in k], key=lambda x: int(x.split("_")[-1])
        )
        image_paths = [(self.image_dir / camera_dict[k]).with_suffix(suffix) for k in _sorted_img_keys]
        assert all(map(lambda x: x.exists(), image_paths)), "Some images are missing!"
        return image_paths

    def _parse_images_from_image_dir(self):
        if self.config.use_ori_res_image:
            image_paths = glob_data(str(self.image_dir / f"*{self.config.image_extension}"))
            assert not self.config.include_mono_prior, "include_mono_prior not supported when use_ori_res_image is True"
            assert (
                self.config.center_crop_type == "no_crop"
            ), "center_crop_type must be no_crop when use_ori_res_image is True"
        else:  # for training w/ mono-priors
            image_paths = glob_data(str(self.config.data / "*_rgb.png"))
        if len(image_paths) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")
        return list(map(Path, image_paths))

    def _load_poses(self, n_images):
        """Load camera poses"""
        camera_dict = np.load(self.camera_anno_path)
        # NOTE: scale_mat_{idx} is always an 4x4 identity matrix for now
        scale_mats = [camera_dict[f"scale_mat_{idx}"].astype(np.float32) for idx in range(n_images)]
        world_mats = [camera_dict[f"world_mat_{idx}"].astype(np.float32) for idx in range(n_images)]

        intrinsics_all = []  # K44
        pose_all = []  # T44_c2w
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)

            if self.config.center_crop_type != "no_crop":
                raise NotImplementedError(f"{self.config.center_crop_type}")

            intrinsics_all.append(torch.from_numpy(intrinsics).float())
            pose_all.append(torch.from_numpy(pose).float())

        fx, fy, cx, cy = [], [], [], []
        all_T44_c2w = []
        for idx in range(n_images):
            intrinsics = intrinsics_all[idx]
            T44_c2w = pose_all[idx]
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            all_T44_c2w.append(T44_c2w)
        fx, fy, cx, cy = map(torch.stack, [fx, fy, cx, cy])
        all_T44_c2w = torch.stack(all_T44_c2w)
        scale_mats = torch.tensor(np.stack(scale_mats[:n_images], 0))

        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        all_T44_c2w[:, 0:3, 1:3] *= -1

        # optionally compute scale factor from object bbox
        if self.config.scale_aabb_to_unit_sphere:
            obj_scale_mat = np.load(self.object_anno_path)["scale_mat_0"].astype(np.float32)
            obj_aabb = load_aabb_from_obj_scale_mat(obj_scale_mat, scale_mats[0])
            obj_diag_len = np.linalg.norm(obj_aabb[1] - obj_aabb[0])
            self.scale_factor = 2 / obj_diag_len
            CONSOLE.print(f"Normalize object aabb into a unit-sphere w/ scale factor={self.scale_factor:.3f}")

        all_T44_c2w[:, :3, 3] *= self.scale_factor
        return scale_mats, all_T44_c2w, fx, fy, cx, cy

    def _build_cameras(self, image_paths, all_T44_c2w, fx, fy, cx, cy):
        # NOTE: assume all cameras share the same intrinsics
        height, width = get_image(image_paths[0]).shape[:2]
        self.image_height, self.image_width = height, width
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=all_T44_c2w[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )
        return cameras

    def _build_scene_box(self, scale_mat: Optional[np.ndarray] = None):
        scene_box_collider_args = {
            "near": self.config.near_far[0] * self.scale_factor,
            "far": self.config.near_far[1] * self.scale_factor,
            "collider_type": self.config.collider_type,
        }

        if scale_mat is None:
            scale_mat = np.eye(4, dtype=np.float32)

        # parse object aabb
        obj_dict = np.load(self.object_anno_path)
        obj_scale_mat = obj_dict["scale_mat_0"].astype(np.float32)

        # build original object bbox
        obj_aabb = load_aabb_from_obj_scale_mat(obj_scale_mat, scale_mat=scale_mat)
        self.obj_aabb = obj_aabb * self.scale_factor

        # build extended & rescaled scene_box
        obj_bbox_len = 1.0 + self.config.object_bounds_extend_ratio
        obj_bbox_min, obj_bbox_max = compute_extended_object_bounds(obj_bbox_len, self.config.object_bounds_extend_mode)

        obj_bbox_min = np.linalg.inv(scale_mat) @ obj_scale_mat @ obj_bbox_min[:, None]
        obj_bbox_max = np.linalg.inv(scale_mat) @ obj_scale_mat @ obj_bbox_max[:, None]
        if (
            obj_bbox_min[:3].min() < -1.0 - 1e-3
            or obj_bbox_max[:3].max() > 1.0 + 1e-3
            or obj_bbox_max[:3].min() < -1.0 - 1e-3
            or obj_bbox_max[:3].max() > 1.0 + 1e-3
        ):
            CONSOLE.log(
                f"[bold red]Object bbox out of bounds (unit-cube): {obj_bbox_min}, {obj_bbox_max}, will clip overflow sides"
            )
        obj_bbox_min = np.clip(obj_bbox_min, -1.0, 1.0)
        obj_bbox_max = np.clip(obj_bbox_max, -1.0, 1.0)

        scene_box_scale_factor = self.config.aabb_scale * self.scale_factor
        if self.config.use_accurate_scene_box:
            _scene_box_str = str(np.stack([obj_bbox_min[:3, 0], obj_bbox_max[:3, 0]], 0).round(2).tolist())
            CONSOLE.log(
                f"Using object aabb as foreground scene box: {_scene_box_str} "
                f"(aabb_scale={self.config.aabb_scale}), scale_factor={self.scale_factor}."
            )
            scene_box = SceneBox(
                aabb=torch.tensor(np.array([obj_bbox_min[:3, 0], obj_bbox_max[:3, 0]]), dtype=torch.float32)
                * scene_box_scale_factor,
                **scene_box_collider_args,
            )
        else:
            CONSOLE.log(
                "Using unit cube as foreground scene box "
                f"scene_box_scale_factor={scene_box_scale_factor} (aabb_scale={self.config.aabb_scale}, scale_factor={self.scale_factor})."
            )
            assert self.config.object_bounds_extend_ratio == 0.0
            assert obj_bbox_min.min() >= -1.0 - 1e-3 and obj_bbox_max.max() <= 1.0 + 1e-3
            scene_box = SceneBox(
                aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32) * scene_box_scale_factor,
                **scene_box_collider_args,
            )
        return scene_box

    def _load_image_features(
        self,
        additional_inputs_dict: Dict[str, Any],
        metadata: Dict[str, Any],
        image_paths: List[Path],
    ) -> None:
        if not self.config.include_image_features:
            return None

        # feat_paths = glob_data(str(self.feat_dir / '*.npz'))
        feat_paths = [self.feat_dir / f"{p.stem}.npz" for p in image_paths]
        if not all([p.exists() for p in feat_paths]):
            raise FileNotFoundError("Missing some feature files!")

        features_np = np.stack(
            [np.load(feat_path)[self.config.feature_key].astype(np.float32) for feat_path in feat_paths], 0  # (h, w, c)
        )  # do not resize feature maps, interpolate with pixel positions
        n, h, w, nc_feat = len(feat_paths), *features_np[0].shape

        _features_flattened = features_np.reshape(-1, nc_feat)  # (n*h*w, c)
        CONSOLE.print(f"Fitting PCA to all image features (#features={len(_features_flattened)})")
        FEATURES_PCA = PCA(n_components=3).fit(_features_flattened)
        _features_pca_flattened = FEATURES_PCA.transform(_features_flattened)
        CONSOLE.print("PCA fitting done!")
        features = torch.tensor(features_np, dtype=torch.float32).permute(0, 3, 1, 2)  # (n, c, h, w)
        # TODO: use (n, h, w, c) and permute when sample features (be consistent with other data terms)

        additional_inputs_dict.update(
            {
                "features": {
                    "func": get_features,
                    "kwargs": {"features": features, "pca": FEATURES_PCA},
                }
            }
        )

        # TODO: save FEATURES_PCA to metadata instead of additional_inputs_dict
        # precompute pca results for all gt features
        features_pca_flattened = torch.tensor(_features_pca_flattened, dtype=torch.float32)
        features_pca_min, features_pca_max = features_pca_flattened.min(0).values, features_pca_flattened.max(0).values
        features_pca = rearrange(features_pca_flattened, "(n h w) c -> n h w c", n=n, h=h, w=w)

        additional_inputs_dict.update(
            {
                "features_pca": {
                    "func": get_features_pca,
                    "kwargs": {"features_pca": features_pca, "min_val": features_pca_min, "max_val": features_pca_max},
                }
            }
        )

    def _load_pointclouds(self, metadata: Dict[str, Any], scale_mat: Optional[np.ndarray] = None):
        """Load and precrocess coarse decomposition results ({fg, bg} {ptcd, feats})"""
        if not self.config.include_coarse_pointclouds:
            return

        if scale_mat is not None:
            raise NotImplementedError
        
        obj_dict = dict(np.load(self.object_anno_path, allow_pickle=True))
        ptcd_data = {}
        if self.config.include_coarse_pointclouds:
            assert set(["fg_pts", "bg_pts", "plane_pts"]) <= set(obj_dict.keys())
            ptcd_data.update(
                {
                    k: torch.tensor(obj_dict[k], dtype=torch.float32) * self.scale_factor
                    for k in ["fg_pts", "bg_pts", "plane_pts"]
                }
            )
            self._load_ptcd_regularization(obj_dict, ptcd_data)

        if self.config.include_coarse_features:
            assert set(["fg_feats", "bg_feats", "plane_feats"]) <= set(obj_dict.keys())
            ptcd_data.update(
                {k: torch.tensor(obj_dict[k], dtype=torch.float32) for k in ["fg_feats", "bg_feats", "plane_feats"]}
            )

        self._downsample_pointclouds(ptcd_data)

        basic_ptcd = BasicPointClouds(**ptcd_data)
        metadata.update({"ptcd_data": basic_ptcd})

    def _load_ptcd_regularization(self, obj_dict: Dict[str, Any], ptcd_data: Dict[str, torch.Tensor]) -> None:
        """load and preprocess data for pointcloud regularization"""
        if self.config.decomposition_mode != "regularization":
            return

        udf_plane_to_fg = torch.tensor(obj_dict["udf_plane_to_fg"], dtype=torch.float32) * self.scale_factor

        fg_sdf_deviation_thr_global = compute_knn_distance_threshold(
            ptcd_data["fg_pts"],
            n_neighbors=self.config.ptcd_reg_fg_global_knn,
            std_ratio=self.config.ptcd_reg_fg_global_std_ratio,
        )

        fg_sdf_deviation_thr_ptwise = compute_ptwise_knn_distance_threshold(
            ptcd_data["fg_pts"], self.config.ptcd_reg_fg_ptwise_knn, self.config.ptcd_reg_fg_ptwise_std_ratio
        )

        # if using a separate plane sdf field
        plane_sdf_deviation_thr_ptwise = compute_ptwise_knn_distance_threshold(
            ptcd_data["plane_pts"], self.config.ptcd_reg_plane_ptwise_knn, self.config.ptcd_reg_plane_ptwise_std_ratio
        )

        # TODO: load plane parameters for random sampling from plane
        ptcd_data.update(
            {
                "udf_plane_to_fg": udf_plane_to_fg,
                "plane_sdf_deviation_thr_ptwise": plane_sdf_deviation_thr_ptwise,
                "fg_sdf_deviation_thr_global": fg_sdf_deviation_thr_global,
                "fg_sdf_deviation_thr_ptwise": fg_sdf_deviation_thr_ptwise,
            }
        )

    def _downsample_pointclouds(self, ptcd_data: Dict[str, torch.Tensor]):
        # TODO: tune the number of fg, bg and plane pts
        # TODO: downsample pointclouds based on feature space distances, instead of euclidean distances
        #       e.g., run k-means and take centroids
        if self.config.downsample_ptcd == -1:
            return

        for partition in ["fg", "bg", "plane"]:
            keys = [
                f"{partition}_pts",
            ]
            if self.config.include_coarse_features:
                keys.append(f"{partition}_feats")
            if partition == "plane" and "udf_plane_to_fg" in ptcd_data:
                keys.append("udf_plane_to_fg")
            pts = ptcd_data[keys[0]]
            n_pts = len(ptcd_data[keys[0]])

            if self.config.downsample_ptcd >= n_pts:
                continue

            CONSOLE.print(f"Downsampling {partition}-ptcd: {n_pts} -> {self.config.downsample_ptcd}")
            down_ratio = self.config.downsample_ptcd / n_pts
            down_inds = fps_torch(pts, ratio=down_ratio, random_start=True)
            for k in keys:
                ptcd_data[k] = ptcd_data[k][down_inds]

    def _load_image_pairs(
        self, image_filenames: List[str], additional_inputs_dict: Dict[str, Any], split: str = "train"
    ):
        if not (self.config.load_pairs and split == "train"):
            return None

        pairs_path = self.config.data / "pairs.txt"
        if not pairs_path.exists():
            raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

        with open(pairs_path, "r") as f:
            pairs = f.readlines()

        def split_ext(x):
            return x.split(".")[0]

        pairs_srcs = []
        for sources_line in pairs:
            sources_array = [int(split_ext(img_name)) for img_name in sources_line.split(" ")]
            if self.config.pairs_sorted_ascending:
                # invert (flip) the source elements s.t. the most similar source is in index 1 (index 0 is reference)
                sources_array = [sources_array[0]] + sources_array[:1:-1]
            pairs_srcs.append(sources_array)
        pairs_srcs = torch.tensor(pairs_srcs)
        all_imgs = torch.stack([get_image(image_filename) for image_filename in sorted(image_filenames)], axis=0).cuda()

        additional_inputs_dict["pairs"] = {
            "func": get_src_from_pairs,
            "kwargs": {
                "all_imgs": all_imgs,
                "pairs_srcs": pairs_srcs,
                "neighbors_num": self.config.neighbors_num,
                "neighbors_shuffle": self.config.neighbors_shuffle,
            },
        }

    def _load_mono_prior(
        self,
        camera_to_worlds: torch.Tensor,
        additional_inputs_dict: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Load monocular depth and normal prior"""
        if not self.config.include_mono_prior:
            return None, None  # depth_images, normal_images

        if self.config.use_ori_res_image:
            raise NotImplementedError("monocular priors are not supported yet when using original resolution images")

        # load monocular depths and normals
        depth_paths = glob_data(str(self.config.data / "*_depth.npy"))
        normal_paths = glob_data(str(self.config.data / "*_normal.npy"))

        depth_images, normal_images = [], []
        for idx, (dpath, npath) in enumerate(zip(depth_paths, normal_paths)):
            depth = np.load(dpath)
            depth_images.append(torch.from_numpy(depth).float())

            normal = np.load(npath)

            # important as the output of omnidata is normalized
            normal = normal * 2.0 - 1.0
            normal = torch.from_numpy(normal).float()

            # transform normal to world coordinate system
            rot = camera_to_worlds[idx][:3, :3].clone()

            normal_map = normal.reshape(3, -1)
            normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

            normal_map = rot @ normal_map
            normal_map = normal_map.permute(1, 0).reshape(*normal.shape[1:], 3)
            normal_images.append(normal_map)

        # stack
        depth_images = torch.stack(depth_images)
        normal_images = torch.stack(normal_images)

        additional_inputs_dict.update(
            {"cues": {"func": get_depths_and_normals, "kwargs": {"depths": depth_images, "normals": normal_images}}}
        )
        return depth_images, normal_images

    def _load_dens3r_prior_without_load(
        self,
        camera_to_worlds: torch.Tensor,
        normal_paths: List[str],
        additional_inputs_dict: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        '''
        only load normal for now
        '''
        normal_exists_flag = True
        normal_filenames = []
        for idx, normal_path in enumerate(normal_paths):
            if not os.path.exists(normal_path):
                normal_exists_flag = False
                break

            normal_filenames.append(normal_path)

        if not normal_exists_flag or len(normal_filenames) == 0:
            return None, None

        CONSOLE.log("load dust3r normal {} image, done.".format(len(normal_filenames)))
        additional_inputs_dict.update(
            {"cues": {"func": load_dens3r_normal, "kwargs": {"normal_filenames": normal_filenames, "camera_to_worlds": camera_to_worlds}}}
        )
        
        return None, None # normal and depth are store in additional_inputs_dict

    def _load_masks(self):
        pass

    def _compute_fg_bbox_masks(
        self,
        all_T44_c2w: List[torch.Tensor],
        all_K33: torch.Tensor,
        image_paths: List[str],
    ) -> Optional[List[Path]]:
        """project 3d bbox to each frame to produce per-frame coarse fg masks.
        Theses masks are useful for weighted ray sampling in foreground and background regions.
        """
        if not self.config.compute_fg_bbox_mask:
            return None

        self.fg_bbox_mask_dir.mkdir(parents=False, exist_ok=True)
        mask_paths = [(self.fg_bbox_mask_dir / Path(p).stem).with_suffix(".png") for p in image_paths]

        if not self.config.force_recompute_fg_bbox_mask and all(p.exists() for p in mask_paths):
            CONSOLE.print("[bold yellow]Using previously saved fg bbox masks.")
            return mask_paths

        IMG_H, IMG_W = self.image_height, self.image_width
        obj_aabb = self.obj_aabb  # (2, 3)
        obj_bbox_pts_2d = project_3d_bbox_to_frames(obj_aabb, all_K33, all_T44_c2w)  # (N, 8, 2)
        obj_bbox_pts_2d = obj_bbox_pts_2d.clamp_min_(0) # for sloving number stability
        obj_polygons = points2polygon(obj_bbox_pts_2d)  # [N, (k, 2)], <x, y>

        for obj_polygon, img_path in tqdm(zip(obj_polygons, image_paths), desc="Computing & saving fg bbox masks"):
            # handle out-of-border pts
            obj_polygon = obj_polygon.round().astype(int)
            min_x, max_x = obj_polygon[:, 0].min(), obj_polygon[:, 0].max()
            min_y, max_y = obj_polygon[:, 1].min(), obj_polygon[:, 1].max()
            offset_x, offset_y = abs(min(0, min_x)), abs(min(0, min_y))
            offset = np.array([[offset_x, offset_y]])
            size_x, size_y = max(max_x - min_x, IMG_W + offset_x), max(max_y - min_y, IMG_H + offset_y)
            _obj_polygon = obj_polygon + offset
            # generate fg mask
            # fg_mask = polygon2mask((size_y, size_x), _obj_polygon[:, [1, 0]])  # (h, w)
            fg_mask = self.polygon2mask_cv(size_y, size_x, _obj_polygon) # faster than polygon2mask
            
            fg_mask = fg_mask[offset_y : offset_y + IMG_H, offset_x : offset_x + IMG_W]
            assert fg_mask.shape == (IMG_H, IMG_W)
            fg_mask_img = fg_mask.astype(np.uint8) * 255
            fg_mask_path = (self.fg_bbox_mask_dir / Path(img_path).stem).with_suffix(".png")
            cv2.imwrite(str(fg_mask_path), fg_mask_img[..., None])
        return mask_paths
    
    def polygon2mask_cv(self, size_y, size_x, obj_polygon_xy):
        # obj_polygon_xy: N×2，(x, y)
        H, W = size_y, size_x
        poly = np.asarray(obj_polygon_xy, dtype=np.float32)
        poly = poly[np.isfinite(poly).all(axis=1)]
        if poly.shape[0] < 3:
            return np.zeros((H, W), dtype=bool)
        poly[:, 0] = np.clip(poly[:, 0], 0, W - 1) # x
        poly[:, 1] = np.clip(poly[:, 1], 0, H - 1) # y
        # 转为 int32
        pts = np.round(poly).astype(np.int32)

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool)

    def _load_segmentation(
        self,
        additional_inputs_dict: Dict[str, Any],
        metadata: Dict[str, Any],
        image_paths: List[Path],
    ) -> None:
        if not self.config.include_segmentation:
            return

        seg_paths = [(self.seg_dir / p.stem).with_suffix(".png") for p in image_paths]
        # seg_masks = [read_mask(p) for p in seg_paths]
        metadata["semantics"] = Semantics(
            filenames=seg_paths, classes=["bg", "fg"], colors=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        )
