from typing import Any, List, Optional, Tuple

import numpy as np
import open3d as o3d
from open3d.geometry import PointCloud
from sklearn.decomposition import PCA

from auto_decomp.utils.geometry.transform import rotmat

from .convert import open3d_ptcd_from_numpy


def compute_knn_distance_threshold(
    points: np.ndarray, kdtree=None, n_neighbors=10, std_ratio=0.5
):
    """compute threshold from global statistic of local distances"""
    # TODO: support both points & pcd
    if kdtree is None:
        pcd = open3d_ptcd_from_numpy(points)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

    mean_dists = []
    for p in points:  # TODO: make parallel
        _, inds, _ = kdtree.search_knn_vector_3d(p, n_neighbors)
        mean_dist = np.linalg.norm(p - points[inds], axis=1, ord=2).mean(0)
        mean_dists.append(mean_dist)
    mean_dists = np.array(mean_dists)
    mean = mean_dists.mean()
    std = np.std(mean_dists)

    max_dist_threshold = mean + std * std_ratio
    return max_dist_threshold


def compute_world_to_object_transformation(
    points: Optional[np.ndarray] = None,
    pcd: Optional[PointCloud] = None,
    ground_dir: np.ndarray = np.array([0.0, 1.0, 0.0]),
    forward_dir: np.ndarray = np.array([1.0, 0.0, 0.0]),  # one of the two axes orthogonal to ground_dir
    axis_length_scaling: int = 4,
    visualize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """
    Returns:
        pca_dirs: object-centric forward direction
        T44_world2object
        traces: plotly traces for visualization
    """
    assert np.sum([o is not None for o in [points, pcd]]) == 1
    xyz = np.array(pcd.points) if pcd is not None else points
    xyz = xyz.copy()  # type: ignore
    pca = PCA(n_components=3)
    center = (xyz.max(0) + xyz.min(0)) / 2
    _ = pca.fit_transform(xyz)
    pca_oris = np.stack(
        [axis_length_scaling * _v * np.sqrt(_l) for _l, _v in zip(pca.explained_variance_, pca.components_)], 0
    )  # pca orientations

    # ground alignment
    y_dirs = np.stack([ground_dir, -ground_dir], 0)  # (2, 3)
    _pca_oris = pca_oris / np.linalg.norm(pca_oris, ord=2, axis=-1, keepdims=True)
    cos_sims = (_pca_oris[:, None] * y_dirs[None]).sum(-1)  # (3, 2)
    _index = np.unravel_index(cos_sims.argmax(), cos_sims.shape)
    src_dir, tgt_dir = _pca_oris[_index[0]], y_dirs[_index[1]]
    R_33_pca2ground = rotmat(src_dir, tgt_dir)
    pca_oris_zero_centered = pca_oris @ R_33_pca2ground.T
    pca_oris = pca_oris_zero_centered + center

    # compute transformation: world -> object-centric
    o_dirs = np.stack([forward_dir, -forward_dir], 0)
    pca_dirs = pca_oris_zero_centered / np.linalg.norm(pca_oris_zero_centered, ord=2, axis=-1, keepdims=True)
    cos_sims = (pca_dirs[:, None] * o_dirs[None]).sum(-1)  # (3, 2)
    _index = np.unravel_index(cos_sims.argmax(), cos_sims.shape)
    src_dir, tgt_dir = pca_dirs[_index[0]], o_dirs[_index[1]]
    R33_world2object = rotmat(src_dir, tgt_dir)
    # translate object center to origin, then rotate
    T44_world2object = np.eye(4)
    T44_world2object[:3, :3] = R33_world2object
    T44_world2object[:3, -1] = -R33_world2object @ center
    if visualize:
        raise NotImplementedError
    else:
        traces = []  # plotly traces
    return pca_dirs, T44_world2object, traces
