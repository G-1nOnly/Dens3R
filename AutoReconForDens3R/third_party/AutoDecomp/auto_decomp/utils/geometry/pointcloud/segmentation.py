"""Pointcloud segmentation methods."""
from collections import deque
from typing import List, Literal, Optional, Tuple

import numpy as np
import open3d as o3d
from loguru import logger
from open3d.geometry import PointCloud

from auto_decomp.utils import tqdm

from .convert import open3d_ptcd_from_numpy
from .plane import fit_plane, Plane
from .misc import compute_knn_distance_threshold


def euclidean_clustering(
    points: np.ndarray,
    radius: Optional[float] = None,  # pre-defined clustering threshold
    n_neighbors: Optional[int] = None,  # compute clustering threshold adaptively
    std_ratio: Optional[float] = None,  # compute clustering threshold adaptively
    verbose: bool = False,
) -> Tuple[List[List[int]], float]:
    """
    Returns:
        clusters (List[List[int]): list of clusters, each represented by a list of point indices
        raidus_threshold (float): the threshold used for clustering
    """
    if points.shape[1] != 3:
        raise NotImplementedError()

    pcd = open3d_ptcd_from_numpy(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    if radius is None:
        assert n_neighbors is not None and std_ratio is not None
        radius = compute_knn_distance_threshold(points, kdtree=pcd_tree, n_neighbors=n_neighbors, std_ratio=std_ratio)
    logger.debug(f"Euclidean clustering radius threshold: {radius}")

    clusters = []
    unprocessed = set(list(range(0, len(points))))
    que = deque()

    pbar = tqdm.tqdm(total=len(points), desc="Euclidean clustering", disable=not verbose)
    while len(unprocessed) > 0 or len(que) > 0:  # TODO: optimize (call PCL-Python?)
        p = unprocessed.pop()
        que.append(p)
        cluster = []
        while len(que) > 0:
            p = que.popleft()
            cluster.append(p)

            _, inds, _ = pcd_tree.search_radius_vector_3d(points[p], radius)

            for new_p in inds:
                if new_p in unprocessed:
                    que.append(new_p)
                    unprocessed.remove(new_p)
            pbar.update(1)
        clusters.append(cluster)
    pbar.close()
    return clusters, radius


def segment_planes(
    points: Optional[np.ndarray] = None,
    pcd: Optional[PointCloud] = None,
    distance_threshold: Optional[float] = None,
    n_ransac_samples: float = 3,
    n_ransac_iters: int = 1000,
    n_neighbors: int = 10,
    std_ratio: float = 1.0,
    min_points_ratio: float = 0.01,
    min_n_consensus: Optional[int] = None,
    enable_euclidean_clustering=False,  # run euclidean clustering before computing the least-square solution.
    least_square: Optional[Literal["OLS", "TLS"]] = None,
    device="cuda",
) -> List[Plane]:
    """Recursively segment all planes from the given pointcloud.
    Assume the given pointcloud is already preprocessed (outlier removal & downsample).

    Args:
        min_points_ratio: recursive segment untill #pts left < len(pcd) * min_points_ratio

    TODO:
        - use LO-RANSAC / GC-RANSAC for better plane fitting
        - current method does not consider the spatial affinity of points belonging to a plane,
          thus points far away from the main plane might be incorrectly treated as plane points
          and affect the plane params significantly.
          - when euclidean_clutering is enabled, this is eased.
    """
    pcd: PointCloud = open3d_ptcd_from_numpy(points) if pcd is None else pcd

    if enable_euclidean_clustering:
        assert least_square is not None

    if distance_threshold is None:
        distance_threshold = compute_knn_distance_threshold(pcd, n_neighbors=n_neighbors, std_ratio=std_ratio)

    n_pts, n_segmented_pts = len(pcd.points), 0
    ori_indices = np.arange(0, n_pts, dtype=int)
    min_n_pts = n_pts - np.floor(n_pts * min_points_ratio)
    planes = []  # store point indices and plane params

    while n_segmented_pts < min_n_pts:
        # TODO: update distance_threshold w/ points left unsegmented?
        if len(pcd.points) < n_ransac_samples:
            break
        plane_model, inlier_inds = pcd.segment_plane(
            distance_threshold=distance_threshold, ransac_n=n_ransac_samples, num_iterations=n_ransac_iters
        )
        if min_n_consensus is not None and len(inlier_inds) < min_n_consensus:
            break

        n_segmented_pts += len(inlier_inds)

        # (optional) least square refinement of plane params
        if least_square is not None and len(inlier_inds) > 0:
            logger.debug(f"Refining plane params w/ {least_square} on inliers.")

            plane_pts = np.array(pcd.select_by_index(inlier_inds).points)

            if enable_euclidean_clustering:  # TODO: make n_neighbors & std_ratio configurable
                clusters, _ = euclidean_clustering(plane_pts, n_neighbors=10, std_ratio=2.0)
                max_cluster_inds = clusters[int(np.argmax([len(c) for c in clusters]))]
                logger.debug(
                    f"Euclidean clustering on inlier plane points: {len(max_cluster_inds)} / {len(plane_pts)} points kept."
                )
                plane_pts = plane_pts[max_cluster_inds]
                inlier_inds = np.array(inlier_inds)[max_cluster_inds].tolist()

            plane_model = fit_plane(plane_pts, method=least_square, device=device)

        planes.append(Plane(ori_indices[inlier_inds], plane_model))

        ori_indices = np.delete(ori_indices, inlier_inds)
        pcd = pcd.select_by_index(inlier_inds, invert=True)

    planes = list(sorted(planes, key=lambda x: len(x.indices), reverse=True))

    return planes
