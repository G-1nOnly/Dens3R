from collections import namedtuple
from typing import Literal, Optional, Tuple

import numpy as np
import pycolmap
import torch
from jaxtyping import Float
from loguru import logger
from skspatial.objects import Plane as SKPlane

from auto_decomp.utils.colmap import read_extrinsics
from auto_decomp.utils.geometry.transform import rotmat

Plane = namedtuple("Plane", ["indices", "params"])


# --------------- plane fitting -------------- #
def fit_plane(points: np.ndarray, method: Literal["OLS", "TLS"] = "OLS", device: str = "cuda"):
    logger.debug(f"Fitting plane w/ {method} on {len(points)} inlier points.")
    if method == "OLS":
        return fit_plane_ols(points)
    elif method == "TLS":
        # return fit_plane_tls(points)  # this is slow
        return fit_plane_tls_torch(points, device=device)
    else:
        ValueError(f"Unknown plane fitting method: {method}")
    logger.debug("Plane fitting done!")


def fit_plane_ols(points: np.ndarray):
    # http://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
    A = np.zeros((3, 3), dtype=np.float32)
    A[2, 2] = len(points)
    A[[0, 1], [0, 1]] = (points**2).sum(0)[:2]
    A[[0, 1], [1, 0]] = (points[:, 0] * points[:, 1]).sum(0)
    A[[0, 2], [2, 0]] = points[:, 0].sum(0)
    A[[1, 2], [2, 1]] = points[:, 1].sum(0)
    b = np.array([(points[:, 0] * points[:, 2]).sum(0), (points[:, 1] * points[:, 2]).sum(0), points[:, 2].sum(0)])
    _params = np.linalg.solve(A, b)  # (3, )
    _normal = np.array([*_params[:2], -1])
    _scale = np.linalg.norm(_normal)
    params = np.concatenate([(_normal / _scale), _params[[-1]] / _scale], 0).astype(np.float32)
    return params  # catersian params (ax + by + cz + d = 0)


def fit_plane_tls(points: np.ndarray):
    """Fit a plane w/ total least sqaure using SVD.

    https://www.ltu.se/cms_fs/1.51590!/svd-fitting.pdf
    https://www.cs.cmu.edu/~16385/s17/Slides/11.5_SVD.pdf
    """
    plane = SKPlane.best_fit(points)
    params = np.array(plane.cartesian())
    _scale = np.linalg.norm(params[:3])
    params = params / _scale
    return params


def fit_plane_tls_torch(points: np.ndarray, device="cuda", max_n_points=20000):
    """
    ref:
        https://github.com/ajhynes7/scikit-spatial/blob/master/src/skspatial/objects/plane.py
        https://github.com/ajhynes7/scikit-spatial/blob/master/src/skspatial/objects/points.py
    """
    if len(points) > max_n_points:  # subsample to avoid OOM
        points = points[np.random.choice(len(points), max_n_points, replace=False)]

    points = torch.tensor(points, dtype=torch.float32, device=device)
    points_centroid = points.mean(0, keepdim=True)
    points_centered = points - points_centroid
    logger.debug(f"TLS SVD input shape: {points_centered.T.shape} (large inputs might cause OOM)")

    # FIXME: limit maximum point numbers to avoid OOM caused by svd
    U, _, _ = torch.linalg.svd(points_centered.T)
    normal = U[:, 2].cpu().numpy()
    centroid = points_centroid[0].cpu().numpy()

    plane = SKPlane(centroid, normal)
    params = np.array(plane.cartesian())
    _scale = np.linalg.norm(params[:3])
    params = params / _scale
    return params


# --------------- utilities -------------- #
def point_to_plane_dist(points: Float[np.ndarray, "N 3"], plane: Float[np.ndarray, "4"], signed: bool = False):
    _normal = plane[None, :3]
    dist = ((_normal * np.array(points)).sum(-1) + plane[-1]) / np.linalg.norm(_normal, ord=2)
    if not signed:
        dist = np.abs(dist)
    return dist


def ground_plane_alignment(
    plane: Plane,
    rec: Optional[pycolmap.Reconstruction] = None,
    avg_camera_loc: Optional[np.ndarray] = None,
    source_vector: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> Tuple[np.ndarray, Plane]:
    """Align the plane with the source_vector, such that the plane normal is consitent with the
    source_vector direction. Infer the positive plane normal direction with various strategies.
    
    Returns:
        R_up (np.ndarray): the rotation matrix transforming plane normal to source_vector
        new_plane (Plane): the aligned plane
    """
    if rec is None and avg_camera_loc is None:
        return ground_plane_alignment_simple(plane, source_vector=source_vector)

    if rec is not None:
        assert avg_camera_loc is None
        # make sure ground plane locates at source_vector(+y) direction w.r.t. cameras
        # 1. make ground normal pointing to +y
        # 2. make cameras locating at -y w.r.t. the ground plane
        all_T44_w2c = read_extrinsics(rec)
        all_T44_c2w = {k: np.linalg.inv(v) for k, v in all_T44_w2c.items()}
        avg_camera_loc = np.stack([v[:3, -1] for v in all_T44_c2w.values()], 0).mean(0)  # (3, )
    else:
        avg_camera_loc = avg_camera_loc

    plane_params = np.array(plane.params, dtype=np.float32)
    plane_normal, plane_dist = plane_params[:3], plane_params[-1]
    norm_normal = np.linalg.norm(plane_normal, ord=2)
    plane_params, plane_normal, plane_dist = map(lambda x: x / norm_normal, [plane_params, plane_normal, plane_dist])

    # make plane_normal pointed to the +y side
    _dot_prod = (plane_normal * source_vector).sum()
    if _dot_prod == 0:
        raise RuntimeError("Direction of the plane normal w.r.t. source_vector is ambiguous.")
    elif _dot_prod < 0:
        plane_params, plane_normal, plane_dist = map(lambda x: -x, [plane_params, plane_normal, plane_dist])

    # make cameras locating at -y w.r.t. the ground plane
    cam2plane_dist = point_to_plane_dist(avg_camera_loc[None], plane_params, signed=True)  # type: ignore

    if cam2plane_dist == 0:
        raise RuntimeError("Averaged camera position locates on the ground plane.")
    elif cam2plane_dist > 0:
        target_vector = -plane_normal
        plane_dist *= -1
    else:
        target_vector = plane_normal

    R_up = np.pad(rotmat(target_vector, source_vector), [0, 1])  # (4, 4)
    R_up[-1, -1] = 1

    new_params = (*source_vector.tolist(), plane_dist)
    new_plane = Plane(indices=plane.indices, params=new_params)
    return R_up, new_plane


def ground_plane_alignment_simple(plane: Plane, source_vector: np.ndarray = np.array([0.0, 1.0, 0.0])):
    """Align the plane with the source_vector, such that the plane normal is consitent with the
    source_vector direction.
    """
    plane_params = np.array(plane.params, dtype=np.float32)
    plane_normal = plane_params[:3]
    norm_normal = np.linalg.norm(plane_normal, ord=2)
    plane_normal = plane_normal / norm_normal

    signs = np.array([1, -1])
    plane_normals = np.stack([plane_normal, plane_normal], 0) * signs[:, None]
    cos_sims = (plane_normals * source_vector[None]).sum(1)
    choice = int(np.argmax(cos_sims))
    target_vector = plane_normals[choice]
    R_up = np.pad(rotmat(target_vector, source_vector), [0, 1])  # (4, 4)
    R_up[-1, -1] = 1

    new_normal = source_vector * signs[choice]
    new_params = (*new_normal.tolist(), plane_params[-1] / norm_normal * signs[choice])
    new_plane = Plane(indices=plane.indices, params=new_params)
    return R_up, new_plane
