""" Preprocess the SfM point cloud"""
import copy
import json
import sys
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import pycolmap
import ray
import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import store
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rich.traceback import install

from auto_decomp.utils import colmap as colmap_utils
from auto_decomp.utils.geometry.pointcloud.pointcloud import NeuralPointCloud
from auto_decomp.utils.misc import EarlyTermination
from auto_decomp.utils.viz_3d import PlotlySaveMode

install(show_locals=False)

PreprocessMode = Enum("PreprocessMode", ["transformer", "ncut"])


@dataclass
class SfMPreprocessConfig:
    mode: PreprocessMode = PreprocessMode.transformer
    """whether preprocess for transformer or ncut"""
    online: bool = False
    """whether preprocess online (return preprocessed data) of offline (save preprocessed data )"""
    data_root: Path = Path("data/")
    inst_rel_dir: Optional[str] = None
    inst_list_path: Optional[Path] = None
    """path to the file containing a list of instances"""
    image_dirname: str = "images"
    feat_dirname: str = "dino_feats_307200"
    feat_dtype: str = "float16"
    exp_dirname: str = "sfm_preprocess"
    n_workers: int = 1
    force_rerun: bool = True
    sfm_dirname: str = "triangulate_loftr-1920000_sequential_np-10"
    load_all_features: bool = True
    """load all features from the provided feature files"""
    cache_name: str = ".cache.json"
    use_cache: bool = True
    """read the latest-produced sfm directory from cache"""
    visualize: bool = True
    """visualize intermediate point clouds during preprocessing"""
    vis_save_mode: PlotlySaveMode = PlotlySaveMode.html

    # filtering configs
    max_reproj_error: Optional[float] = None
    """if None, compute filtering threshold adaptively"""
    min_track_length: Optional[float] = None
    """if None, compute filtering threshold adaptively"""
    stat_outlier_n_neighbors: int = 30
    """statistical outlier removal"""
    stat_outlier_std_ratio: float = 0.5
    """statistical outlier removal"""

    # TODO: update other preprocess configs for NCut preprocessing


sfm_preprocess_store = store(group="decomp/sfm_preprocess")
sfm_preprocess_store = sfm_preprocess_store(SfMPreprocessConfig, name="base")
store.add_to_hydra_store()


class SfMPreprocessWorker:
    def __init__(self, cfg: DictConfig, inst_rel_dir: str, device: str = "cuda"):
        """Preprocess a single SfM point cloud."""
        self.cfg = cfg
        self.device = device
        self._init_paths(inst_rel_dir)
        self.rec = pycolmap.Reconstruction(self.sfm_dir)
        self._sanity_check()
        early_stop = self._handle_resume()
        if early_stop:
            return

        logger.info(f"Processing instance: {self.inst_rel_dir}")
        self.update_thresholds()

    def _init_paths(self, inst_rel_dir: str):
        cfg = self.cfg
        self.inst_rel_dir = inst_rel_dir
        inst_root = cfg.data_root / self.inst_rel_dir
        cfg.cache_path = inst_root / cfg.cache_name  # caches storing the sfm paths (.cache.json)

        if cfg.use_cache:
            with open(cfg.cache_path, "r") as f:
                cache = json.load(f)
            sfm_dir = Path(cache["sfm_dir"])
            sfm_root = inst_root / sfm_dir.parent.relative_to(inst_root)
            feat_dirname = cache["feat_dirname"]
            logger.info(
                f"Reading directories from cache: SfM:{sfm_dir.relative_to(inst_root)} | Feature:{feat_dirname}"
            )
        else:
            sfm_root = inst_root / cfg.sfm_dirname
            sfm_dir = sfm_root / "sfm"
            feat_dirname = cfg.feat_dirname
        manhattan_sfm_dir = sfm_root / "manhattan"
        if (manhattan_sfm_dir / "points3D.bin").exists():
            sfm_dir = manhattan_sfm_dir

        self.sfm_root, self.sfm_dir = sfm_root, sfm_dir
        self.images_dir = inst_root / cfg.image_dirname
        self.exp_dir = self.sfm_root / cfg.exp_dirname
        self.feat_dir = inst_root / feat_dirname
        self.vis_dir = self.exp_dir / "vis_plots"

        # mkdir
        self.exp_dir.mkdir(exist_ok=True)
        self.vis_dir.mkdir(exist_ok=True)

    def _handle_resume(self) -> bool:
        early_stop = False
        if not self.cfg.force_rerun:
            # TODO: check if this instance is already processed.
            raise NotImplementedError
        else:
            early_stop = False
        return early_stop

    def _sanity_check(self) -> None:
        # check sfm model
        _required_fns = ["cameras.bin", "images.bin", "points3D.bin"]
        if not all([(self.sfm_dir / fn).exists() for fn in _required_fns]):
            raise EarlyTermination(
                f"Incomplete SfM model ({self.sfm_dir}), " f'at least one of ({" ".join(_required_fns)}) is missing!'
            )

        # check vit features
        if not self.feat_dir.exists():
            raise EarlyTermination(f"feat_dir does not exist: {self.feat_dir}")
        else:  # check if features are extracted for all registered images
            for img_id in self.rec.reg_image_ids():
                img_name = self.rec.images[img_id].name
                feat_path = (self.feat_dir / img_name).with_suffix(".npz")
                if not feat_path.exists():
                    # logger.error(f'Cannot find the required feature file: {str(feat_path)} (for image "{img_name}")')
                    raise EarlyTermination(
                        f'Cannot find the required feature file: {str(feat_path)} (for image "{img_name}")'
                    )

    def update_thresholds(self):
        """Update preprocessing thresholds adaptively"""

        # max_reproj_error & min_track_length of the SfM point cloud
        if self.cfg.max_reproj_error is None or self.cfg.min_track_length is None:
            # NOTE: the distri. of min_track_lengths is left-skewed (mostly a Pareto distri.)
            #       the distri. of reproj_errors is approximately gaussian
            bbs = self.rec.compute_bounding_box(0.001, 0.999)  # filtering extreme outliers
            p3Ds = {
                p3D_id: p3D
                for p3D_id, p3D in self.rec.points3D.items()
                if (p3D.xyz >= bbs[0]).all() and (p3D.xyz <= bbs[1]).all()
            }
            reproj_errors = [p3D.error for p3D in p3Ds.values()]
            track_lengths = [p3D.track.length() for p3D in p3Ds.values()]
            min_track_length = np.median(track_lengths)  # TODO: use a more statistically sound indicator
            _reproj_error_std_ratio = 3.0
            max_reproj_error = np.mean(reproj_errors) + np.std(reproj_errors) * _reproj_error_std_ratio
            if self.cfg.max_reproj_error is None:
                self.cfg.max_reproj_error = float(max_reproj_error)
            if self.cfg.min_track_length is None:
                self.cfg.min_track_length = float(min_track_length)
            logger.info(f"Using adaptive thresholds: {max_reproj_error=} | {min_track_length=}")

    def __call__(self) -> Optional[Dict[str, Any]]:
        """Perform a series of pre-processing on the SfM Point Cloud.

        1. filtering w/ reproj-error & track-length
        2. statistical outlier removal -> SfM Segmentation Transformer
        3. downsampling w/ voxelization
        4. filtering points behind most of the cameras
        5. filtering part of the ground points -> NCut Segmentation
        """
        data = {"inst_name": self.inst_rel_dir.replace("/", "_")}
        self.parse_poses(data)
        self.build_neural_ptcd(data)
        self.stat_outlier_removal(data)
        if self.cfg.mode == PreprocessMode.transformer:
            if self.cfg.online:
                return data  # "poses", "neural_ptcd"
            else:
                self.save_transformer_inference()

        self.voxel_downsample(data)
        self.filter_pts_behind_cameras(data)
        self.filter_ground_plane_pts(data)
        if self.cfg.online:
            return data
        else:
            self.save_ncut_inference()

    def parse_poses(self, data):
        data.update({"poses": colmap_utils.parse_poses(self.rec)})

    def build_neural_ptcd(self, data):
        """Build neural point cloud with basic preprocessing based on track-length and reproj-error"""
        neural_ptcd = NeuralPointCloud.from_pycolmap_rec(
            self.rec,
            self.images_dir,
            self.feat_dir,
            self.cfg.max_reproj_error,
            self.cfg.min_track_length,
            self.device,
            feature_dtype=self.cfg.feat_dtype,
            load_all_features=self.cfg.load_all_features,
            vis_dir=self.vis_dir,
            plotly_save_mode=self.cfg.vis_save_mode,
        )
        if self.cfg.visualize:
            neural_ptcd.visualize_ptcd("step-1")
        data.update({"neural_ptcd": neural_ptcd})

    def stat_outlier_removal(self, data):
        data["neural_ptcd"].statistical_outlier_removal(
            self.cfg.stat_outlier_n_neighbors, self.cfg.stat_outlier_std_ratio
        )
        # TODO: vis & logging

    def voxel_downsample(self, data):
        raise NotImplementedError

    def filter_pts_behind_cameras(self, data):
        raise NotImplementedError

    def filter_ground_plane_pts(self, data):
        raise NotImplementedError

    def save_transformer_inference(self):
        """Save preprocessed data for transformer inference"""
        # TODO: tensor -> ndarray
        # TODO: save feature with neural_ptcd.feature_dtype
        raise NotImplementedError

    def save_ncut_inference(self):
        """Save preprocessed data for transformer inference"""
        # TODO: save feature with neural_ptcd.feature_dtype
        raise NotImplementedError


# TODO: Use a reusable Actor
@ray.remote(num_cpus=1, num_gpus=1)
def ray_wrapper(cfg, task_id=None, **kwargs) -> Tuple[bool, str, Optional[str]]:
    torch.cuda.empty_cache()
    gpu_id = ray.get_gpu_ids()[0]
    inst_rel_dir = kwargs["inst_rel_dir"]

    logger.configure(extra={"gpu_id": gpu_id, "task_id": task_id, "inst": inst_rel_dir})

    try:
        SfMPreprocessWorker(cfg, **kwargs)()
        logger.info("SfM preprocessing done!")
        return True, inst_rel_dir, None
    except (EarlyTermination, RuntimeError, AssertionError) as err:
        err_str = traceback.format_exc()
        logger.error(f"SfM preprocessing failed: {err_str}")
        return False, inst_rel_dir, err_str


class SfMPreprocess:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        OmegaConf.set_struct(cfg, False)
        self.setup_logger()
        if self.cfg.n_workers > 1:
            ray.init(num_cpus=self.cfg.n_workers, num_gpus=self.cfg.n_workers)

    def __call__(self) -> Optional[Dict[str, Any]]:
        assert np.array([i is not None for i in [self.cfg.inst_rel_dir, self.cfg.inst_list_path]]).sum() == 1

        # Run SfM preprocessing
        if self.cfg.inst_rel_dir is not None:
            assert self.cfg.n_workers == 1
            torch.cuda.empty_cache()
            logger.configure(extra={"gpu_id": 0, "task_id": 0, "inst": self.cfg.inst_rel_dir})
            return SfMPreprocessWorker(self.cfg, self.cfg.inst_rel_dir, device="cuda")()
        else:
            assert not self.cfg.online, "Online mode is not supported for multiple sequences!"
            with open(self.cfg.inst_list_path, "r") as f:
                inst_rel_dirs = f.read().strip().split()
            obj_refs = [
                ray_wrapper.remote(copy.deepcopy(self.cfg), task_id=task_id, inst_rel_dir=inst_rel_dir)  # type: ignore
                for task_id, inst_rel_dir in enumerate(inst_rel_dirs)
            ]
            logger.info(f"{len(obj_refs)} sequences to be processed!")
            ret_vals = ray.get(obj_refs)

            # Log results
            ret_states = np.array([s for s, _, _ in ret_vals], dtype=int)
            insts = np.array([i for _, i, _ in ret_vals])
            errors = np.array([e for _, _, e in ret_vals])
            n_success = ret_states.sum()
            failed_insts = insts[~(ret_states.astype(bool))]
            failed_errors = errors[~(ret_states.astype(bool))]
            logger.info(f"Successfully executed tasks: {n_success} / {len(ret_states)}")
            if n_success != len(ret_states):
                err_str = "\n".join([f"{_id}: {_err}" for _id, _err in zip(failed_insts, failed_errors)])
                logger.error(f"Failed instances: \n{err_str}")

    def setup_logger(self):
        logger_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "- <level>{message}</level>"
        )
        logger.remove()
        logger.add(sys.stderr, format=logger_format)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=SfMPreprocessConfig)
    hydra.main(config_name="config", version_base=None)(lambda cfg: SfMPreprocess(cfg)())()
