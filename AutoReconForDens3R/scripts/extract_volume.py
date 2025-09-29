#!/usr/bin/env python
"""
Extract volumes and save them as .mrc files.
"""
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import mrcfile
import numpy as np
import torch
import tyro
from rich.console import Console
from tqdm import tqdm
from typing_extensions import Literal

from nerfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)


@dataclass
class ExtractVolume:
    """Load a checkpoint, extract volume data, and save it to a mrc file."""

    load_config: Path
    """Path to config YAML file."""
    resolution: int = 512
    dynamic_resolution: bool = True
    output_path: Path = Path("volume_data.mrc")
    side_chunk_size: int = 128
    volume_type: Literal["sdf", "density", "segmentation"] = "sdf"
    # TODO: support feature volume extraction (after PCA)
    use_train_scene_box: bool = True
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    seg_aware_sdf: bool = True

    # TODO: support config override (manually override for now)
    downsample_ptcd: int = 1000
    feature_seg_knn: int = 10
    feature_seg_dist_weighted_avg: bool = False
    feature_seg_separate_bg_plane: bool = False

    def main(self) -> None:
        assert self.output_path.suffix == ".mrc"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        _, pipeline, _ = eval_setup(self.load_config, cache_images=False, config_override_fn=self._config_override_fn)
        if self.use_train_scene_box:
            aabb = pipeline.datamanager.train_dataset.scene_box.aabb
            self.bounding_box_min = aabb[0].cpu().numpy()
            self.bounding_box_max = aabb[1].cpu().numpy()
            _bbox_str = str(np.stack([self.bounding_box_min, self.bounding_box_max], 0).round(2))
            CONSOLE.print(f"Using training SceneBox for mesh extraction: {_bbox_str}")

        CONSOLE.print("Extract mesh with marching cubes and may take a while")

        volume_extractor = partial(
            extract_volume,
            resolution=self.resolution,
            dynamic_resolution=self.dynamic_resolution,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            output_path=self.output_path,
            side_chunk_size=self.side_chunk_size,
        )
        if self.volume_type == "density":
            raise NotImplementedError
        elif self.volume_type == "sdf":
            self.extract_sdf_volume(pipeline, volume_extractor)
        elif self.volume_type == "segmentation":
            self.extract_seg_volume(pipeline, volume_extractor)
        else:
            raise ValueError(f"Unknown volume_type: {self.volume_type}")

    def _config_override_fn(self, config):
        """override config fields manually"""
        if hasattr(config.pipeline.model, "feature_seg_field"):
            config.pipeline.model.feature_seg_field.knn = self.feature_seg_knn
            config.pipeline.model.feature_seg_field.distance_weighted_average = self.feature_seg_dist_weighted_avg
            config.pipeline.model.feature_seg_field.separate_bg_plane = self.feature_seg_separate_bg_plane
        config.pipeline.datamanager.dataparser.downsample_ptcd = self.downsample_ptcd
        return config

    def extract_seg_volume(self, pipeline, volume_extractor):
        seg_fn = pipeline.model.query_seg_field
        return volume_extractor(seg_fn, volume_type="segmentation")

    def extract_sdf_volume(self, pipeline, volume_extractor):
        if self.seg_aware_sdf:
            sdf_fn = pipeline.model.seg_aware_sdf
        else:

            def sdf_fn(x):
                return pipeline.model.field.forward_geonetwork(x)[:, [0]]

        return volume_extractor(sdf_fn, volume_type="sdf")


@torch.no_grad()
def extract_volume(
    field_fn,  # fn: (n, 3) -> (n, 1)
    resolution=512,  # resolution of the longest side
    dynamic_resolution=False,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    output_path: Optional[Path] = None,
    side_chunk_size: int = 128,
    volume_type: Literal["sdf", "density", "segmentation"] = "sdf",
):
    N = side_chunk_size

    grid_min, grid_max = bounding_box_min, bounding_box_max
    # dynamically scale resolution of each dimension w.r.t. bb size
    side_lengths = grid_max - grid_min
    if dynamic_resolution:
        resolutions = np.round(side_lengths / side_lengths.max() * resolution).astype(int)
    else:
        resolutions = [
            resolution,
        ] * 3

    X = torch.linspace(grid_min[0], grid_max[0], resolutions[0]).cuda().split(N)
    Y = torch.linspace(grid_min[1], grid_max[1], resolutions[1]).cuda().split(N)
    Z = torch.linspace(grid_min[2], grid_max[2], resolutions[2]).cuda().split(N)
    n_chunks = len(X) * len(Y) * len(Z)
    volume = np.zeros(resolutions, dtype=np.float32)

    # FIXME: the generated coords are incorrect? (compare the dumped volume with MC mesh)
    with tqdm(total=n_chunks, desc="Chunked volume extraction") as pbar:
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    # val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    field_val = field_fn(pts).detach().cpu().numpy()
                    field_val = field_val.reshape(len(xs), len(ys), len(zs))
                    volume[xi * N : xi * N + len(xs), yi * N : yi * N + len(ys), zi * N : zi * N + len(zs)] = field_val
                    pbar.update(1)

    # Save volume as .mrc
    # pad the border of the extracted volume
    mrc_volume = postprocess_volume_for_mrc(volume, volume_type)

    if output_path is not None:
        with mrcfile.new_mmap(str(output_path), overwrite=True, shape=mrc_volume.shape, mrc_mode=2) as mrc:
            mrc.data[:] = mrc_volume
        CONSOLE.print(f".mrc volume saved: {output_path}")

    return volume


def postprocess_volume_for_mrc(volume, volume_type):
    resolutions = volume.shape
    pads, resos = [int(10 * r / 256) for r in resolutions], resolutions

    volume, pad_val = compute_volume_and_pad_val(volume, volume_type)
    padded_volume_sizes = [r + 2 * pad for r, pad in zip(resos, pads)]
    padded_volume = np.full(padded_volume_sizes, pad_val, dtype=float)
    padded_volume[pads[0] : -pads[0], pads[1] : -pads[1], pads[2] : -pads[2]] = volume
    return padded_volume


def compute_volume_and_pad_val(volume: np.ndarray, volume_type: str):
    if volume_type == "sdf":  # chimerax donnot support sdf, so we transform it into "density-alike"
        pad_val = -np.abs(volume).max() * 2
        volume = -volume
    elif volume_type == "density":
        pad_val = np.abs(volume).max() * 2
        volume = volume
    elif volume_type == "segmentation":  # assume fg:1 | bg:0
        pad_val = 0
        volume = volume
    else:
        raise ValueError(f"Unknown volume_type: {volume_type}")
    return volume, pad_val


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[ExtractVolume]).main()


if __name__ == "__main__":
    entrypoint()
