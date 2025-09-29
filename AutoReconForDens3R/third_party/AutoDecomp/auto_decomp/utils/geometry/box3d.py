from typing import Literal, Tuple, Union

import numpy as np
import open3d as o3d
import torch
from matplotlib.axis import Axis
from open3d.geometry import AxisAlignedBoundingBox
from torch import Tensor

from auto_decomp.utils.geometry.pointcloud import Plane, point_to_plane_dist

try:
    from pytorch3d.ops import iou_box3d
except ImportError as ie:
    print(ie)

"""
The 'vertices' format follows PyTorch3D's definition

        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)

Orientation of the underlying coordinate frame is arbitrary. (colmap is used here)
(the strict order of vertices is not meaningful, only the relative order matter)
"""


def min_max_to_vertices_torch(box: Tensor) -> Tensor:
    # box: (B, 2, 3)
    B = box.shape[0]
    _min, _max = box[:, 0], box[:, 1]  # (B, 3)
    assert (_max >= _min).all()
    vertices = box.new_ones((B, 8, 3))
    vertices[:, [0, 3, 4, 7], 0] = _min[:, 0]
    vertices[:, [1, 2, 5, 6], 0] = _max[:, 0]
    vertices[:, [0, 1, 4, 5], 1] = _min[:, 1]
    vertices[:, [2, 3, 6, 7], 1] = _max[:, 1]
    vertices[:, [0, 1, 2, 3], 2] = _min[:, 2]
    vertices[:, [4, 5, 6, 7], 2] = _max[:, 2]
    return vertices  # (B, 8, 3)


def min_max_to_vertices(box):
    _min, _max = box[0], box[1]
    assert (_max >= _min).all()
    vertices = np.ones_like(box, shape=(8, 3))
    vertices[[0, 3, 4, 7], 0] = _min[0]
    vertices[[1, 2, 5, 6], 0] = _max[0]
    vertices[[0, 1, 4, 5], 1] = _min[1]
    vertices[[2, 3, 6, 7], 1] = _max[1]
    vertices[[0, 1, 2, 3], 2] = _min[2]
    vertices[[4, 5, 6, 7], 2] = _max[2]
    return vertices


def _vertices_to_min_max(box):
    # box: (8, 3)
    return np.stack([box.min(0), box.max(0)])


def _center_extent_to_vertices(box):  # extent:=half_extent
    box_min_max = np.stack([box[0] - box[1], box[0] + box[1]], 0)  # (2, 3)
    return min_max_to_vertices(box_min_max)


def _vertices_to_center_extent(box):
    _min, _max = box.min(0), box.max(0)
    center = (_min + _max) / 2
    half_extent = (_max - _min) / 2
    return np.stack([center, half_extent], 0)


def _center_extent_to_min_max(box):
    center, extent = box[0], box[1]
    min_corner, max_corner = center - extent, center + extent
    box = np.stack([min_corner, max_corner], 0)
    return box


BOX_FORMAT_CONVERTOR = {
    "min_max": {"vertices": min_max_to_vertices},
    "center_extent": {
        "vertices": _center_extent_to_vertices,
        "min_max": _center_extent_to_min_max,
    },
    "vertices": {"min_max": _vertices_to_min_max, "center_extent": _vertices_to_center_extent},
}


def convert_box_format(box, src_format, tgt_format):
    # TODO: implement a pytorch version
    if src_format not in BOX_FORMAT_CONVERTOR:
        raise ValueError(f"Unknown {src_format=}")
    if tgt_format not in BOX_FORMAT_CONVERTOR:
        raise ValueError(f"Unknown {tgt_format=}")

    convertor = BOX_FORMAT_CONVERTOR[src_format].get(
        tgt_format,
        lambda x: BOX_FORMAT_CONVERTOR["vertices"][tgt_format](BOX_FORMAT_CONVERTOR[src_format]["vertices"](x)),
    )
    return convertor(box)


def compute_box3d_iou(
    box0: np.ndarray,
    box1: np.ndarray,
    box_format: str = "min_max",
    eps: float = 1e-4,
    return_volume: bool = False,
    device: str = "cpu",
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Compute IoU b/w two 3d boxes.

    Args:
        box{0, 1}: (2, 3) or (8, 3)
        return_volume: return the intersection volume b/w two boxes.
    """
    box0, box1 = map(lambda x: convert_box_format(x, box_format, "vertices"), [box0, box1])
    _box0, _box1 = map(lambda x: torch.tensor(x, device=device, dtype=torch.float32)[None], [box0, box1])  # (1, 8, 3)

    intersect_vol, iou = iou_box3d.box3d_overlap(_box0, _box1, eps=eps)
    intersect_vol, iou = intersect_vol.cpu().numpy()[0, 0], iou.cpu().numpy()[0, 0]
    return (iou, intersect_vol) if return_volume else iou


def normalize(array: np.ndarray, axis: int = -1):
    norm = np.linalg.norm(array, ord=2, axis=axis, keepdims=True)
    array = array / norm
    return array


def extend_aabb(
    aabb: AxisAlignedBoundingBox,
    ground_plane: Plane,
    ground_dir: np.ndarray = np.array([0.0, 1.0, 0.0]),
    ratio: float = 0.05,
    extend_bottom: bool = False,
    extend_reference: Literal["before", "after"] = "after",
) -> AxisAlignedBoundingBox:
    """Extend the bottom side of an aabb to the ground_plane, and enlarge
    spatial extents of other sides. (assume world up-direction equals to the ground_plane normal.)

    Args:
        extend_bottom (bool): if True, extend the bottom side (near to ground) of the aabb as well
        extend_renference: when computing the enlargement of the aabb, whether use the aabb before
            or after it is aligned with the ground_plane
    """
    normal = np.array(ground_plane.params[:3])
    normal = normal / np.linalg.norm(normal, ord=2)
    assert np.allclose(normal, ground_dir) or np.allclose(normal, -ground_dir)
    center, half_extent = aabb.get_center(), aabb.get_half_extent()
    enlargement = half_extent * ratio
    side_points = np.stack([center + normal * half_extent, center - normal * half_extent], 0)

    # choose the side_point nearest to the ground plane
    signed_dists = point_to_plane_dist(side_points, np.array(ground_plane.params), signed=True)
    choice = int(np.abs(signed_dists).argmin())

    # determine the bottom->ground direction
    mult_signed_dists = signed_dists[0] * signed_dists[1]
    if mult_signed_dists == 0:  # bbox collapses to ground plane
        direction = None
    elif mult_signed_dists > 0:  # on the same side
        direction = side_points[choice] - side_points[1 - choice]
    else:  # on different sides
        direction = side_points[1 - choice] - side_points[choice]

    # align the bottom plane w/ ground plane
    direction = normalize(direction) if direction is not None else np.zeros((3,), dtype=side_points.dtype)
    bottom_point = side_points[choice] + direction * np.abs(signed_dists[choice])
    up_point = side_points[1 - choice]

    # for some corner cases, we only got partial segmentation, then use "after" is more appropriate
    if extend_reference == "after":
        enlargement[1] = (ratio * np.abs(up_point - bottom_point) / 2)[1]
    else:
        assert extend_reference == "before"
    
    up_point = up_point + normalize(up_point - bottom_point, axis=-1) * enlargement  # only enlarge the upper plane
    if extend_bottom:
        bottom_point = bottom_point + normalize(bottom_point - up_point, axis=-1) * enlargement
    center = (up_point + bottom_point) / 2

    _half_extent = np.abs(up_point - bottom_point) / 2
    half_extent = half_extent + (1 - ground_dir) * enlargement
    half_extent = np.where(np.isclose(_half_extent, np.zeros_like(_half_extent)), half_extent, _half_extent)
    min_bound, max_bound = center - half_extent, center + half_extent

    new_aabb = AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    return new_aabb


if __name__ == "__main__":
    box0 = np.array([[-1, -1, -1], [1, 1, 1]], dtype=np.float32)
    box1 = np.array([[-2, -2, -2], [3, 4, 5]], dtype=np.float32)

    # test converter
    assert (convert_box_format(convert_box_format(box0, "min_max", "vertices"), "vertices", "min_max") == box0).all()
    assert (convert_box_format(convert_box_format(box1, "min_max", "vertices"), "vertices", "min_max") == box1).all()

    # test compute_box3d_iou
    iou, vol = compute_box3d_iou(box0, box1, box_format="min_max", return_volume=True)
    assert np.allclose(vol, 8) and np.allclose(iou, 8 / (5 * 6 * 7))
