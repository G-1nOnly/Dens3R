from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import torch
from einops import repeat
from hydra_zen import builds, store
from jaxtyping import Float
from loguru import logger
from pytorch3d.ops import knn_points
from torch import Tensor

from auto_decomp.utils.geometry.box3d import (
    extend_aabb,
    min_max_to_vertices,
    min_max_to_vertices_torch,
)
from auto_decomp.utils.geometry.pointcloud import (
    compute_world_to_object_transformation,  # TODO: compute_oriented_aabb -> .
)
from auto_decomp.utils.geometry.pointcloud import (
    Plane,
    compute_knn_distance_threshold,
    euclidean_clustering,
    ground_plane_alignment,
    open3d_ptcd_from_numpy,
    point_to_plane_dist,
)
from auto_decomp.utils.geometry.pointcloud import segment_planes as segment_all_planes


# --------------- helper functions -------------- #
def _to_numpy(x: Union[np.ndarray, Tensor]) -> Tuple[np.ndarray, Any]:
    if isinstance(x, Tensor):
        device = x.device  # pyright: ignore
        _x = x.cpu().numpy()
    else:
        device = None
        _x = x
    return _x, device


def _to_tensor(x: Union[np.ndarray, Tensor], device: str) -> Tensor:
    return torch.tensor(x, device=device) if isinstance(x, np.ndarray) else x


def extract_max_cluster(
    x: Tensor,
    n_neighbors: int = 10,
    std_ratio: float = 2.0,
    return_ndarray: bool = False,
    return_inds: bool = True,
):
    # TODO: implement a native torch version
    _x = x.cpu().numpy()  # type: np.ndarray
    clusters, radius = euclidean_clustering(_x, n_neighbors=n_neighbors, std_ratio=std_ratio)
    max_cluster_idx = int(np.argmax([len(c) for c in clusters]))
    cluster = clusters[max_cluster_idx]  # ndarray
    _x = _x[cluster]

    ret_x = x.new_tensor(_x) if not return_ndarray else _x
    ret_vals = [
        ret_x,
    ]
    if return_inds:
        ret_inds = x.new_tensor(cluster, dtype=torch.int32) if not return_ndarray else cluster
        ret_vals.append(ret_inds)
    return ret_vals if len(ret_vals) > 0 else ret_vals[0]


def remove_outliers(
    x: Union[np.ndarray, torch.Tensor],
    n_neighbors: int = 10,
    std_ratio: float = 2.0,
    return_ndarray: bool = False,
    return_inds: bool = True,
    device: Optional[str] = None,
):
    _x, device = _to_numpy(x)

    pcd = open3d_ptcd_from_numpy(_x)
    _, inlier_idxs = pcd.remove_statistical_outlier(nb_neighbors=n_neighbors, std_ratio=std_ratio)
    _x = _x[inlier_idxs]

    ret_x = _x if return_ndarray else torch.tensor(_x, device=device)
    ret_vals = [
        ret_x,
    ]
    if return_inds:
        ret_inds = torch.tensor(inlier_idxs, device=device, dtype=torch.int32)
        ret_vals.append(ret_inds)
    return ret_vals if len(ret_vals) > 0 else ret_vals[0]


def segment_fg_supporting_plane(
    fg_pts: Union[np.ndarray, torch.Tensor],  # (N0, 3)
    bg_pts: Union[np.ndarray, torch.Tensor],  # (N1, 3)
    device: Optional[str] = None,
    n_point_samples: int = 6,  # number of samples for ransac
    n_ransac_iters: int = 10000,
    n_neighbors: int = 10,
    std_ratio: float = 1.0,  # FIXME: the segmented plane might truncate the fg object
    min_stop_points_ratio: float = 0.1,
    min_n_consensus: Optional[int] = None,
    avg_top_k_ratio: float = 0.01,
    min_fg_plane_points_ratio: float = 0.05,
    max_points_to_plane_dist_ratio: float = 0.2,
    max_n_underground_points_ratio: float = 0.1,
    max_bg_pt_dist_ratio: float = 1.0,  # ignore bg pts whose min distance to fg pts > ratio * fg extent
    enable_euclidean_clustering: bool = True,
    plane_ls_refinement: Literal["TLS", "OLS"] = "TLS",
):
    """Segment the plane supporting the fg pointcloud (if existed).

    TODO:
        - encapsulate into a class
        - estimate gravity direction w/ camera poses.
        - increase min_stop_points_ratio! (for faster computation)
        - increase n_point_samples
        - set min_n_consensus
    """
    # build separate fg ptcd & bg ptcd
    if len(bg_pts) == 0 or len(fg_pts) == 0:
        logger.warning(f"Invalid number of bg / fg points: #bg={len(bg_pts)} | #fg={len(fg_pts)}")
        return None

    bg_pts_ori_inds = np.arange(len(bg_pts))

    # (optional) filter bg_pts
    filter_bg_pts = max_bg_pt_dist_ratio is not None
    if filter_bg_pts:  # filter out bg_pts far away from the fg object
        fg_pts, bg_pts = map(lambda x: _to_tensor(x, device), (fg_pts, bg_pts))
        fg_extent = fg_pts.max(0).values - fg_pts.min(0).values
        _dist_thr = fg_extent.max() * max_bg_pt_dist_ratio
        min_bg_to_fg_dists = torch.sqrt(knn_points(bg_pts[None], fg_pts[None], K=1).dists[0, :, 0])  # (N1, )
        _bg_kept_mask = min_bg_to_fg_dists <= _dist_thr
        bg_pts = bg_pts[_bg_kept_mask]
        bg_pts_ori_inds = bg_pts_ori_inds[_bg_kept_mask.cpu().numpy()]
        logger.debug(f"#bg_pts kept: {_bg_kept_mask.sum()}/{len(_bg_kept_mask)}")
        if len(bg_pts) < n_point_samples:
            logger.warning(f"Not enough bg_pts left after filtering: {len(bg_pts)}/{n_point_samples}")
            return None

    fg_pts, _device = _to_numpy(fg_pts)
    bg_pts, _device = _to_numpy(bg_pts)

    # segment all planes from the bg ptcd
    # TODO: use LO-RANSAC (since distance threshold varies in different positions)
    distance_thr_fg, distance_thr_bg = map(
        lambda x: compute_knn_distance_threshold(x, n_neighbors=n_neighbors, std_ratio=std_ratio),
        (fg_pts, bg_pts),
    )
    planes = segment_all_planes(
        points=bg_pts,
        n_ransac_samples=n_point_samples,
        n_ransac_iters=n_ransac_iters,
        distance_threshold=distance_thr_bg if filter_bg_pts else distance_thr_fg,
        min_points_ratio=min_stop_points_ratio,
        min_n_consensus=min_n_consensus,
        enable_euclidean_clustering=enable_euclidean_clustering,
        least_square=plane_ls_refinement,
        device=_device,
    )

    # select the supporting ground plane with heuristics
    # -- 1. ground should be near to fg pts
    # -- 2. ground plane shoud locate on one side of the fg pts
    # -- 3. keep the plane w/ maximum number of pts
    n_fg_pts = len(fg_pts)
    max_n_underground_points = n_fg_pts * max_n_underground_points_ratio
    _top_k_avg = int(n_fg_pts * avg_top_k_ratio)
    min_n_plane_pts = int(n_fg_pts * min_fg_plane_points_ratio)
    fg_max_extent = (np.max(fg_pts, 0) - np.min(fg_pts, 0)).max()
    max_pt_to_plane_dist = fg_max_extent * max_points_to_plane_dist_ratio
    fg_to_plane_dists, n_plane_pts, min_n_pts_both_sides = [], [], []
    for plane in planes:  # TODO: make parallel
        _normal = np.array(plane.params)[None, :3]
        pts_to_plane_dists = point_to_plane_dist(fg_pts, np.array(plane.params), signed=True)

        # bipartite points to two sides of the plane
        vec_plane_to_pts = pts_to_plane_dists[:, None] * _normal
        on_side_one = (vec_plane_to_pts * _normal).sum(-1) > 0
        n_on_side_one = on_side_one.sum()
        min_both_side = min(n_on_side_one, len(on_side_one) - n_on_side_one)
        min_n_pts_both_sides.append(min_both_side)

        # mean points-to-plane distances
        abs_pts_to_plane_dists = np.abs(pts_to_plane_dists)
        topk_inds = np.argpartition(abs_pts_to_plane_dists, _top_k_avg, axis=0)[:_top_k_avg]
        avg_min_dist = abs_pts_to_plane_dists[topk_inds].mean(0)
        fg_to_plane_dists.append(avg_min_dist)
        n_plane_pts.append(len(plane.indices))
    fg_to_plane_dists, n_plane_pts = np.array(fg_to_plane_dists), np.array(n_plane_pts)
    min_n_pts_both_sides = np.array(min_n_pts_both_sides)

    plane_inds = np.arange(len(planes))
    _kept_mask = (
        (n_plane_pts >= min_n_plane_pts)
        & (fg_to_plane_dists <= max_pt_to_plane_dist)
        & (min_n_pts_both_sides <= max_n_underground_points)
    )
    if not _kept_mask.any():
        # if no plane is detected at all, keep one plane at least
        logger.warning("No valid plane detected! Try keeping the plane with maximum number of points.")
        if len(plane_inds) > 0:
            plane = planes[0]
            plane = Plane(params=plane.params, indices=bg_pts_ori_inds[plane.indices])
            return plane
        else:
            # return None
            raise RuntimeError("No valid plane detected!")
    plane_ind = int(plane_inds[_kept_mask][0])
    plane = planes[plane_ind]
    plane = Plane(params=plane.params, indices=bg_pts_ori_inds[plane.indices])
    return plane


def extract_object_centric_fg_aabb(
    pts: Float[Tensor, "n_pts 3"],
    T44_w2o: Float[Tensor, "4 4"],
    support_plane: Plane,
    align_with_plane: bool = True,
    extend_ratio: float = 0.05,
    extend_bottom: bool = False,
) -> Tuple[Float[Tensor, "2 3"], Plane]:
    """Extract the foreground axis-aligned bounding box in the object-centric frame.

    NOTE: we assume pts coordinate frame is already aligned with support_plane.
    """
    if len(pts) == 0:
        return pts.new_zeros((2, 3))

    # apply transformation to (ground-aligned) object-centric frame
    pts = pts @ T44_w2o[:3, :3].T + T44_w2o[:3, -1]
    aabb_min_max = torch.stack([pts.min(0).values, pts.max(0).values], 0)

    # extend aabb to align with the estimated ground plane
    if align_with_plane:
        old_aabb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=aabb_min_max[0].cpu().numpy(), max_bound=aabb_min_max[1].cpu().numpy()
        )
        # also apply transformation to the plane
        # only translate the plane (since we assume pts coordinates are already aligned with support_plane)
        plane_params = np.array(support_plane.params)
        plane_params[-1] -= T44_w2o[1, -1]
        support_plane = Plane(support_plane.indices, plane_params)
        new_aabb = extend_aabb(
            old_aabb,
            support_plane,
            ground_dir=np.array([0.0, 1.0, 0.0]),
            ratio=extend_ratio,
            extend_bottom=extend_bottom,
        )
        new_center, new_half_extent = new_aabb.get_center(), new_aabb.get_half_extent()
        new_min, new_max = new_center - new_half_extent, new_center + new_half_extent
        aabb_min_max = aabb_min_max.new_tensor(np.stack([new_min, new_max], 0))
    return aabb_min_max, support_plane


# --------------- postprocessing -------------- #
@dataclass
class PostprocessConfig:
    threshold: float = 0.5
    # fg ptcd euclidean clustering params
    enable_euc_cluster: bool = False
    euc_cluster_n_neighbors: int = 20
    euc_cluster_std_ratio: float = 2.0
    # fg ptcd outlier removal params
    enable_outlier_removal: bool = False
    outlier_removal_n_neighbors: int = 10
    outlier_removal_std_ratio: float = 2.0
    # fg bbox extraction params
    enable_plane_alignment: bool = False
    """Estimate parameters the fg object's supporting plane, which can be used for
    plane-aligned PCA for fg pts reorientaion."""
    extend_bbox_ratio: float = 0.10
    extend_bbox_bottom_plane: bool = False
    align_bbox_with_plane: bool = False
    reorient_bbox_with_pca: bool = False  # TODO: rename: enable_object_centric_transform
    return_full_decomposition: bool = False
    """compute and return full decomposition results (besides the fg bbox)"""


postprocess_store_train = store(group="decomp/transformer/postprocess_train")
postprocess_store_train(PostprocessConfig, name="base")

postprocess_store_test = store(group="decomp/transformer/postprocess_test")
postprocess_store_test(PostprocessConfig, name="eval")  # TODO: used for evaluation, consistent with GT
postprocess_store_test(
    PostprocessConfig,
    name="inference",
    enable_euc_cluster=True,
    enable_outlier_removal=True,
    enable_plane_alignment=True,
    extend_bbox_bottom_plane=True,
    align_bbox_with_plane=True,
    reorient_bbox_with_pca=True,
    return_full_decomposition=True,
)
postprocess_store_test(
    PostprocessConfig,
    name="inference_cvpr",
    enable_euc_cluster=True,
    enable_outlier_removal=True,
    enable_plane_alignment=True,
    extend_bbox_bottom_plane=True,
    align_bbox_with_plane=True,
    reorient_bbox_with_pca=True,
    return_full_decomposition=True,
    extend_bbox_ratio=0.15
)

store.add_to_hydra_store()


class Postprocess:
    def __init__(self, cfg: PostprocessConfig):
        """Postprocess the raw sfm segmentation results, extract the oriented aabb
        and compute object-centric transformation.
        """
        self.cfg = cfg

    def __call__(
        self,
        x: Float[Tensor, "batch n_pts 3"],
        prob: Float[Tensor, "batch n_pts 1"],
        avg_cam_loc: Optional[Float[Tensor, "3"]] = None,
        aux_data: Optional[Dict[str, Tensor]] = None,
        rot_mat: Optional[Float[Tensor, "batch 3 3"]] = None,
        instance_names: Optional[Tensor] = None,
    ) -> Tuple[Float[Tensor, "batch n_pts 3"], Float[Tensor, "batch n_pts 3"], Dict[str, Dict[str, Any]]]:
        """Run postprocessing, extract aabb and build decomposition results for regularizing neus training.
        Args:
            x: the SfM pointcloud
            prob: fg segmentation probabilities
            avg_cam_loc: The average camera location in world space, used for determing the up-direction of the plane.
                # FIXME: only support in testing with batch_size = 1 for now.
            rot_mat: the random rotation matrix applied during training as random augmentation.
                This is used for transforming fg pts to the input object-centric frame for aabb extraction.
            aux_data: auxiliary attributes associated with points (e.g., point-wise features)
        Returns:
            fg_aabb_verts_no_postprocess: Float[Tensor, "batch n_pts 3"]
            fg_aabb_verts: Float[Tensor, "batch n_pts 3"]
            full_decomp_results: Dict[str, Dict[str, Any]]
        """
        # preparation
        _b, _dtype, _device = x.shape[0], x.dtype, x.device
        x, prob = x.detach(), prob.detach()
        if aux_data is not None:
            aux_data: Dict[str, Float[Tensor, "batch n_pts n_f"]] = {k: v.detach() for k, v in aux_data.items()}

        if (
            rot_mat is not None
        ):  # transform to object-centric frame (assume the input world frame during training is object-centric)
            x = torch.einsum("bac,bnc->bna", rot_mat.transpose(1, 2), x)
        if avg_cam_loc is not None:
            assert _b == 1
            avg_cam_loc = avg_cam_loc.cpu().numpy()
        if instance_names is not None:
            instance_names = [str(n) for n in instance_names]  # type: ignore

        fg_mask = (prob >= self.cfg.threshold)[..., 0]  # (B, N)
        fg_inds = [torch.nonzero(m)[:, 0].cpu().numpy() for m in fg_mask]  # [B, (N, )]
        fg_x = [x[m] for x, m in zip(x, fg_mask)]  # [B, (N, 3)]
        bg_x = [x[~m] for x, m in zip(x, fg_mask)]  # [B, (N, 3)]
        bg_inds = [torch.nonzero(~m)[:, 0].cpu().numpy() for m in fg_mask]  # [B, (N, )]
        plane_inds = [x.copy() for x in bg_inds]  # [B, (N, )]

        # register attributes
        self.B, self.dtype, self.device = _b, _dtype, _device
        self.x, self.prob, self.fg_x, self.bg_x = x, prob, fg_x, bg_x
        self.fg_inds, self.bg_inds, self.plane_inds = fg_inds, bg_inds, plane_inds  # List[np.ndarray]
        self.rot_mat, self.avg_cam_loc, self.instance_names = rot_mat, avg_cam_loc, instance_names
        self.aux_data = aux_data

        # option-1: extract bbox w/o postprocessing
        all_fg_aabb_verts_no_postprocess = self.extract_bbox_no_postprocess(fg_x, rot_mat=rot_mat)
        if not (self.cfg.enable_euc_cluster or self.cfg.enable_outlier_removal):
            if self.reorient_bbox_with_pca:
                raise NotImplementedError()
            return all_fg_aabb_verts_no_postprocess, None, None

        # option-2: extract bbox w/ postprocessing
        # 1. euclidean clustering
        self.run_euclidean_clustering()
        # 2. outlier_removal
        self.run_outlier_removal()
        # 3. plane_alignment
        self.run_plane_alignment()
        # 4. re-orient bbox with pca
        self.compute_object_centric_transformation()
        self.fg_x = [torch.tensor(_x, dtype=_dtype, device=_device) for _x in self.fg_x]
        # 5. extract aabb
        self.build_fg_aabb()
        # 6. compute alignment results for further ues
        all_decomp_results = self.compute_full_decomposition()
        return all_fg_aabb_verts_no_postprocess, self.all_fg_aabb_verts, all_decomp_results

    def run_euclidean_clustering(self):
        if not self.cfg.enable_euc_cluster:
            return

        if self.B > 1:
            raise NotImplementedError  # TODO: Joblib
        # TODO: add a wrapper for sequential processing
        fg_x = [
            extract_max_cluster(
                _x,
                n_neighbors=self.cfg.euc_cluster_n_neighbors,
                std_ratio=self.cfg.euc_cluster_std_ratio,
                return_ndarray=True,
                return_inds=True,
            )
            if len(_x) > 0
            else _x
            for _x in self.fg_x
        ]  # List[List[np.ndarray, np.ndarray]]
        _fg_inds = [_inds for _, _inds in fg_x]
        self.fg_inds = [old_inds[new_inds] for old_inds, new_inds in zip(self.fg_inds, _fg_inds)]
        self.fg_x = [_x for _x, _ in fg_x]

    def run_outlier_removal(self):
        if not self.cfg.enable_outlier_removal:
            return

        if self.B > 1:
            raise NotImplementedError  # TODO: Joblib
        fg_x = [
            remove_outliers(
                _x,
                n_neighbors=self.cfg.outlier_removal_n_neighbors,
                std_ratio=self.cfg.outlier_removal_std_ratio,
                return_ndarray=self.cfg.enable_plane_alignment,
                return_inds=True,
                device=self.device,
            )
            if len(_x) > 0
            else _x
            for _x in self.fg_x
        ]  # List[Union[torch.Tensor, np.ndarray]]
        _fg_inds = [_inds for _x, _inds in fg_x]
        self.fg_inds = [old_inds[new_inds] for old_inds, new_inds in zip(self.fg_inds, _fg_inds)]
        self.fg_x = [_x for _x, _inds in fg_x]

    def run_plane_alignment(self):
        if not self.cfg.enable_plane_alignment:
            self.fg_x_gnd_aligned = [None for _ in self.fg_x]
            self.planes = [None for _ in self.fg_x]
            self.all_R44_up = [np.eye(4) for _ in self.fg_x]  # useless
            return

        if self.B > 1:
            raise NotImplementedError  # TODO: Joblib

        self._segment_support_plane()
        self._align_world_with_plane()

    def _segment_support_plane(self):
        planes = [
            segment_fg_supporting_plane(
                _fg_pts,
                _bg_pts,
                device=self.device,
                # instance_name=self.instance_names[_b_id]
                # if self.instance_names is not None
                # else None,  # pyright: ignore
            )
            for _b_id, (_fg_pts, _bg_pts) in enumerate(zip(self.fg_x, self.bg_x))
        ]
        self.planes = planes

    def _align_world_with_plane(self):
        """Align the up-dir of world with the ground plane normal."""
        all_R44_up, planes = map(
            list,
            zip(
                *[
                    ground_plane_alignment(plane, rec=None, avg_camera_loc=self.avg_cam_loc)
                    if plane is not None
                    else (np.eye(4), None)
                    for _b_id, plane in enumerate(self.planes)
                ]
            ),
        )
        fg_x_gnd_aligned = [fg_pts @ R_up[:3, :3].T for R_up, fg_pts in zip(all_R44_up, self.fg_x)]  # type: ignore
        self.all_R44_up = all_R44_up
        self.fg_x_gnd_aligned = fg_x_gnd_aligned
        self.planes = planes  # planes in the new world frame

    def compute_object_centric_transformation(self):
        """Transform fg point cloud to the object-centric coordinate system with
        object intrinsic orientation estimated by PCA.
        """
        if not self.cfg.reorient_bbox_with_pca:
            self.all_T44_w2o = repeat(torch.eye(4, dtype=self.dtype, device=self.device), "r w -> b r w", b=self.B)
            return
        if not self.cfg.enable_plane_alignment:
            # in this case, we should pass ground_dir & forward_dir to computer_world_to_object_transformation
            raise NotImplementedError

        all_fg_pts = self.fg_x_gnd_aligned if self.cfg.enable_plane_alignment else self.fg_x
        all_pca_dirs, all_T44_w2o, _ = list(
            zip(
                *[
                    compute_world_to_object_transformation(points=fg_pts)
                    if len(fg_pts) > 3
                    else (None, np.eye(4), None)
                    for fg_pts in all_fg_pts
                ]
            )
        )
        all_T44_w2o = [T44_w2o @ R44_up for R44_up, T44_w2o in zip(self.all_R44_up, all_T44_w2o)]
        self.all_T44_w2o = torch.tensor(np.stack(all_T44_w2o, 0), dtype=self.dtype, device=self.device)  # (B, 4, 4)

    def build_fg_aabb(self):
        """Transform fg points to (plane-aligned) object-centric frame and take the fg aabb,
        then transform the fg aabb back to the world frame.
        """
        if not self.cfg.enable_plane_alignment:
            # currently, we assume the fg points are already aligned with the supporting plane
            raise NotImplementedError

        # extract fg aabb (aligned with the supporting plane)
        all_fg_aabb_min_max, all_fg_aabb_verts, planes_obj = self._extract_object_centric_fg_aabb()
        self.all_fg_aabb_min_max = all_fg_aabb_min_max  # fg aabb in object-centric frame

        # object frame -> world frame
        all_R33_w2o, all_t31_w2o = self.all_T44_w2o[:, :3, :3], self.all_T44_w2o[:, :3, [-1]]
        all_R33_o2w = all_R33_w2o.transpose(1, 2)
        all_t31_o2w = -all_R33_o2w @ all_t31_w2o
        all_fg_aabb_verts = torch.einsum("bac,bnc->bna", all_R33_o2w, all_fg_aabb_verts) + all_t31_o2w.transpose(1, 2)

        # world frame -> augmented world frame (random aug during training)
        if self.rot_mat is not None:  # transform back to augmented world frame
            all_fg_aabb_verts = torch.einsum("bac,bnc->bna", self.rot_mat, all_fg_aabb_verts)
        self.all_fg_aabb_verts = all_fg_aabb_verts

    def _extract_object_centric_fg_aabb(self) -> Float[Tensor, "batch 8 3"]:
        """Transform fg points to (plane-aligned) object-centric frame and take the fg aabb."""
        all_fg_aabb_min_max, planes_obj = [], []
        for _b_id, _pts in enumerate(self.fg_x):
            aabb_min_max, plane_obj = extract_object_centric_fg_aabb(
                _pts,
                self.all_T44_w2o[_b_id],
                self.planes[_b_id],
                align_with_plane=self.cfg.enable_plane_alignment,
                extend_ratio=self.cfg.extend_bbox_ratio,
                extend_bottom=self.cfg.extend_bbox_bottom_plane,
            )
            all_fg_aabb_min_max.append(aabb_min_max)
            planes_obj.append(plane_obj)  # object-centric plane
        all_fg_aabb_verts = min_max_to_vertices_torch(torch.stack(all_fg_aabb_min_max, 0))  # (B, 8, 3)
        return all_fg_aabb_min_max, all_fg_aabb_verts, planes_obj

    def extract_bbox_no_postprocess(
        self, fg_x: List[Float[Tensor, "n_pts 3"]], rot_mat: Optional[Float[Tensor, "batch 3 3"]] = None
    ) -> Tuple[Float[Tensor, "batch n_pts 3"], None, None]:
        fg_bbox_min_max = []
        for _pts in fg_x:
            if len(_pts) == 0:
                _pts = self.x.new_zeros((2, 3))
            fg_bbox_min_max.append(torch.stack([_pts.min(0).values, _pts.max(0).values], 0))
        fg_bbox_verts = min_max_to_vertices_torch(torch.stack(fg_bbox_min_max, 0))

        if rot_mat is not None:  # transform back to the augmented frame (random aug)
            fg_bbox_verts = torch.einsum("bac,bnc->bna", rot_mat, fg_bbox_verts)
        return fg_bbox_verts, None, None

    def compute_full_decomposition(self) -> Dict[str, Dict[str, Any]]:
        """Compute decomposition results for neus supervision.
        Returns:
            Dict[str, Dict[str, Any]]: {
                "{instance-name-1}": {
                    "T44_w2o": (4, 4), normalized-world => ground-aligned object-centric frame
                    "aabb_min_max": (2, 3)
                    "fg_pts": (n_fg, 3)
                    "bg_pts": (n_bg, 3)
                    "plane_pts": (n_plane, 3)
                    "udf_plane_to_fg": (n_plane,)
                    "plane_params": (4,)
                    "aux_fg": Dict[str, (n_fg, ...)]
                    "aux_bg": Dict[str, (n_bg, ...)]
                    "aux_plane": Dict[str, (n_plane, ...)]
                }
            }
        """
        # sanity check
        assert self.cfg.enable_plane_alignment
        # augmented_world -> original_world (cannot handle random aug during training)
        assert self.rot_mat is None or (self.rot_mat.cpu().numpy() == np.eye(3)[None]).all()
        assert self.instance_names is not None

        all_decomp_results = {}
        for b_id in range(self.B):
            decomp_results = {}  # instance-name -> decomp results
            # build decomposition results for supervising neus
            self._build_decomposed_pts(b_id, decomp_results)

            # decompose auxiliary attributes (e.g., point-wise features)
            self._build_decomposed_aux_attrs(b_id, decomp_results)

            all_decomp_results[self.instance_names[b_id]] = decomp_results
        return all_decomp_results
    
    def set_bg_in_bbox_to_fg(self, fg_pts, bg_pts, fg_aabb_min_max_info):
        '''
        fg_aabb_min_max_info: (2, 3)
        fg_pts: N, 3
        bg_pts: M, 3
        '''
        aabb_min = fg_aabb_min_max_info.min(dim=0).values.to(bg_pts.device)
        aabb_max = fg_aabb_min_max_info.max(dim=0).values.to(bg_pts.device)
        eps = 1e-6
        ge_min = bg_pts >= (aabb_min - eps)
        le_max = bg_pts <= (aabb_max + eps)
        inside_mask = (ge_min & le_max).all(dim=-1)  # (M,)

        bg_in = bg_pts[inside_mask]
        fg_pts_updated = torch.cat([fg_pts, bg_in], dim=0)
        if (~inside_mask).sum() > 0:
            bg_pts_update = bg_pts[~inside_mask]
        else:
            bg_pts_update = bg_pts.new_zeros((0, 3))

        return fg_pts_updated, bg_pts_update
    
    def _build_decomposed_pts(self, b_id: int, decomp_results: Dict[str, Any]):
        """Build decomposed pointcloud for neus supervision."""
        # 1. extract pts for neus supervision (fg_pts, bg_pts, plane_pts, plane_params)
        fg_pts, bg_pts = self.fg_x[b_id], self.bg_x[b_id]  # world-frame pts
        if self.planes[b_id] is not None:
            # NOTE: plane normal is already in object-centric frame (if world frame is plane-aligned), but translation hasn't yet
            plane_params = np.array(self.planes[b_id].params)  # type: ignore
            _plane_inds = bg_pts.new_tensor(self.planes[b_id].indices, dtype=torch.long)  # type: ignore
            _bg_kept_mask = bg_pts.new_ones((len(bg_pts),), dtype=torch.bool)
            _bg_kept_mask[_plane_inds] = 0
            plane_pts = bg_pts[_plane_inds]
            bg_pts = bg_pts[_bg_kept_mask]  # the plane is noisy, some pts around it are kept in bg_pts
            self.plane_inds[b_id] = self.plane_inds[b_id][_plane_inds.cpu().numpy()]
            self.bg_inds[b_id] = self.bg_inds[b_id][_bg_kept_mask.cpu().numpy()]
        else:
            plane_params, plane_pts = None, bg_pts.new_zeros((0, 3))
            self.plane_inds[b_id] = np.zeros((0,), dtype=int)
        
        # set bg_pts in bbox to fg_pts
        bbox_x = self.all_fg_aabb_verts[b_id][:, 0] # [N]
        bbox_y = self.all_fg_aabb_verts[b_id][:, 1]
        bbox_z = self.all_fg_aabb_verts[b_id][:, 2]
        aabb_tmp = torch.tensor([[bbox_x.min(), bbox_y.min(), bbox_z.min()],
                                [bbox_x.max(), bbox_y.max(), bbox_z.max()]])
        fg_pts, bg_pts = self.set_bg_in_bbox_to_fg(fg_pts, bg_pts, aabb_tmp)
        
        logger.debug(
            f"Decomposition results: #fg_pts={len(fg_pts)} | #bg_pts={len(bg_pts)} | #plane_pts={len(plane_pts)}"
        )

        # 2. transform all pts to object-centric frame
        T44_w2o: Float[Tensor, "4 4"] = self.all_T44_w2o[b_id]
        fg_pts, bg_pts, plane_pts = map(
            lambda pts: pts @ T44_w2o[:3, :3].T + T44_w2o[:3, -1], [fg_pts, bg_pts, plane_pts]
        )
        T44_w2o = T44_w2o.cpu().numpy()
        if plane_params is not None:
            plane_params[-1] -= T44_w2o[1, -1]

        # 3. compute approximate SDF for plane_pts
        if len(plane_pts) > 0:  # TODO: consider more fg_pts for better approximation
            udf_plane_to_fg = torch.sqrt(
                knn_points(plane_pts[None], fg_pts[None], K=1).dists[0, :, 0]
            )  # (n_plane_pts, )
        else:
            udf_plane_to_fg = bg_pts.new_zeros((0,))
        decomp_results.update(
            {
                "T44_w2o": T44_w2o,  # (4, 4), normalized-world => ground-aligned object-centric frame
                "aabb_min_max": self.all_fg_aabb_min_max[b_id].cpu().numpy(),  # (2, 3)
                "fg_pts": fg_pts.cpu().numpy(),  # (n_fg, 3)
                "bg_pts": bg_pts.cpu().numpy(),  # (n_bg, 3)
                "plane_pts": plane_pts.cpu().numpy(),  # (n_plane, 3)
                "udf_plane_to_fg": udf_plane_to_fg.cpu().numpy(),  # (n_plane,)
                "plane_params": plane_params,  # (4,)
            }
        )

    def _build_decomposed_aux_attrs(self, b_id: int, decomp_results: Dict[str, Any]):
        """Decompose auxiliary attributes associated with points (e.g., point-wise features)"""
        if self.aux_data is None:
            return

        aux_data: Dict[str, Float[Tensor, "batch n_pts n_f"]] = self.aux_data
        fg_inds, bg_inds, plane_inds = self.fg_inds[b_id], self.bg_inds[b_id], self.plane_inds[b_id]
        assert len(fg_inds) == len(decomp_results["fg_pts"])
        aux_data_fg: Dict[str, Float[Tensor, "n_pts n_f"]] = {
            k: v[b_id].cpu().numpy()[fg_inds] for k, v in aux_data.items()
        }
        aux_data_bg = {k: v[b_id].cpu().numpy()[bg_inds] for k, v in aux_data.items()}
        aux_data_plane = {k: v[b_id].cpu().numpy()[plane_inds] for k, v in aux_data.items()}
        decomp_results.update({"aux_fg": aux_data_fg, "aux_bg": aux_data_bg, "aux_plane": aux_data_plane})
