
from typing import Literal, Optional
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from functools import partial
from copy import deepcopy
from scipy import cluster

import torch
import tyro
import pymeshlab
import numpy as np
import open3d as o3d
from rich.console import Console


CONSOLE = Console()


@dataclass
class PreprocessNeusPose:
    data_root: Path  # e.g., /data/haotong/0013_01_0075
    camera_filename: str = "cameras_sphere.npz"
    image_dirname: str = "image"
    fg_aabb_extend_ratio: float = 0.15
    
    def main(self):
        """ transform neus poses w/ additional fg bbox annotation to the autorecon format """
        self.camera_path = self.data_root / self.camera_filename
        self.save_camera_path = self.data_root / "cameras_norm-obj-side-2.0.npz"
        self.save_object_path = self.data_root / "objects_norm-obj-side-2.0.npz"
        self.image_dir = self.data_root / self.image_dirname
        image_paths = list((self.data_root / self.image_dirname).iterdir())
        _suffix = image_paths[0].suffix
        assert all([p.suffix == _suffix for p in image_paths])
        self.n_images =  len(image_paths)
        
        # load camera dict
        camera_dict = self.load_camera_dict()
        
        # renormalize scene
        
        # save camera dict & object dict
    
    def load_camera_dict(self):
        camera_dict = np.load(self.camera_path)
        scale_mats = [camera_dict[f"scale_mat_{idx}"].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict[f"world_mat_{idx}"].astype(np.float32) for idx in range(self.n_images)]
        fg_aabb = camera_dict["bbox"]  # (2, 3), [min, max]
        return {
            'scale_mats': scale_mats, 'world_mats': world_mats, 'fg_aabb': fg_aabb
        }

    def renormalize_scene(self, camera_dict):
        """
        extend the fg_aabb w/ a ratio and normalize it into a unit-cube
        
        the fg aabb is inaccurate whatsoever...
        """
        pass

    def _parse_images_from_camera_dict(self):
        """Parse images from camera dict, which is useful when there are too many frames.
        """
        camera_dict = np.load(self.camera_path)
        suffix = next(self.image_dir.iterdir()).suffix
        _sorted_img_keys = sorted([k for k in camera_dict.keys() if 'image_name' in k],
                                  key=lambda x: int(x.split('_')[-1]))
        image_paths = [(self.image_dir / camera_dict[k]).with_suffix(suffix) for k in _sorted_img_keys]
        assert all(map(lambda x: x.exists(), image_paths)), 'Some images are missing!'
        return image_paths
