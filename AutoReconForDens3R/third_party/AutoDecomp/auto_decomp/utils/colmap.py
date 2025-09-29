import numpy as np
import pycolmap


def read_extrinsics(rec: pycolmap.Reconstruction):
    all_T44_w2c = {}
    bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
    for img_id in rec.reg_image_ids():
        img = rec.images[img_id]
        img_name = img.name
        T44_w2c = np.concatenate([np.concatenate([img.rotmat(), img.tvec[:, None]], 1), bottom], 0)
        all_T44_w2c[img_name] = T44_w2c
    return all_T44_w2c


def parse_poses(rec):
    """Parse all registered poses from a reconstruction.

    Returns:
        Dict: {image_name: {'T44_w2c': np.ndarray, 'K33': np.ndarray}}
    """
    all_poses = {}
    all_T44_w2c = read_extrinsics(rec)
    all_K33 = {}  # assume pinhole camera
    for img_id in rec.reg_image_ids():
        img = rec.images[img_id]
        cam = rec.cameras[img.camera_id]
        img_name = img.name
        K33 = cam.calibration_matrix()
        all_K33[img_name] = K33
    for img_name in all_T44_w2c.keys():
        all_poses[img_name] = {"T44_w2c": all_T44_w2c[img_name], "K33": all_K33[img_name]}
    return all_poses
