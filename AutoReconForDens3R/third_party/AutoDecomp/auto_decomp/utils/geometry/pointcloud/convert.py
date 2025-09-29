from typing import Optional

import numpy as np
import open3d as o3d


def open3d_ptcd_from_numpy(
    points: np.ndarray, colors: Optional[np.ndarray] = None, normals: Optional[np.ndarray] = None
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        assert colors.shape[0] == points.shape[0]
        if colors.max() > 1 or colors.dtype == np.uint8:
            colors = colors.astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        assert normals.shape[0] == points.shape[0]
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd
