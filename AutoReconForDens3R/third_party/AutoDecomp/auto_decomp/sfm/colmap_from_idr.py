"""
Build an empty colmap mode with camera poses from annotations in the IDR format.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import hydra
import imagesize
import numpy as np
from hloc.utils.read_write_model import Camera, Image, rotmat2qvec, write_model
from hydra.core.config_store import ConfigStore
from hydra_zen import store
from loguru import logger
from tqdm import tqdm


@dataclass
class IDR2COLMAPConfig:
    """Convert data in the IDR format to COLMAP models."""

    data_root: Path = Path("data")
    inst_rel_dir: Optional[str] = None
    pose_filename: str = "cameras.npz"
    image_filename: str = "images"
    save_dirname: str = "sfm_from_idr"


idr2colmap_store = store(group="idr2colmap")
idr2colmap_store(IDR2COLMAPConfig, name="base")


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
    pose[:3, 3] = (t[:3] / t[3])[:, 0]  # merged transformation

    return intrinsics, pose  # cam2world psoe


def parse_idr_poses(pose_path, image_dir) -> Dict[str, Dict[str, np.ndarray]]:
    cameras = np.load(pose_path)
    poses = {}
    frame_ids = sorted(list(set([int(n.split("_")[-1]) for n in cameras.keys() if "bbox" not in n])))
    image_paths = sorted(image_dir.iterdir())
    for f_id, image_path in zip(frame_ids, image_paths):
        world_mat, scale_mat = cameras[f"world_mat_{f_id}"], cameras[f"scale_mat_{f_id}"]
        world_mat = world_mat @ scale_mat
        K44_cv, T44_c2w_cv = load_K_Rt_from_P(None, world_mat[:3, :])
        width, height = imagesize.get(str(image_path))
        poses[image_path.name] = {
            "frame_id": f_id,
            "K44_cv": K44_cv,
            "T44_c2w_cv": T44_c2w_cv,
            "width": width,
            "height": height,
        }
    return poses


def process_instance(cfg, inst_dirname):
    inst_dir = cfg.data_root / inst_dirname
    logger.info(f"Processing instance: {inst_dir}")
    image_dir = inst_dir / cfg.image_filename
    pose_path = inst_dir / cfg.pose_filename
    sfm_dir = inst_dir / cfg.save_dirname
    sfm_dir.mkdir(parents=False, exist_ok=True)

    # parse colmap poses from neus poses
    poses = parse_idr_poses(pose_path, image_dir)

    # build colmap model
    # NOTE: assuming all intrinsics are the same
    cameras, images = {}, {}
    for img_name, frame in poses.items():
        frame_id = int(frame["frame_id"])
        # create cameras
        cam_params = frame["K44_cv"][[0, 1, 0, 1], [0, 1, 2, 2]].tolist()
        camera = Camera(id=frame_id, model="PINHOLE", width=frame["width"], height=frame["height"], params=cam_params)
        cameras[frame_id] = camera
        # create images
        R33_c2w, t3_c2w = frame["T44_c2w_cv"][:3, :3], frame["T44_c2w_cv"][:3, -1]
        R33_c2w = R33_c2w.T
        t3_w2c = -(R33_c2w @ t3_c2w)
        image = Image(
            id=frame_id,
            qvec=rotmat2qvec(R33_c2w),
            tvec=t3_w2c,
            camera_id=frame_id,
            name=img_name,
            xys=np.zeros((0, 2), float),
            point3D_ids=np.full(0, -1, int),
        )
        images[frame_id] = image
    write_model(cameras, images, {}, path=str(sfm_dir), ext=".bin")


def main(idr2colmap: IDR2COLMAPConfig):
    cfg = idr2colmap

    if cfg.inst_rel_dir is not None:
        process_instance(cfg, cfg.inst_rel_dir)
    else:
        for inst_dir in tqdm(cfg.data_root.iterdir(), desc="Converting data (in IDR format) to COLMAP models"):
            if not (inst_dir / cfg.pose_filename).exists():
                continue
            process_instance(cfg, inst_dir.stem)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=IDR2COLMAPConfig)
    hydra.main(config_name="config", version_base=None)(main)()
