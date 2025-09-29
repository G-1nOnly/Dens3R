#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations
import shutil
from collections import defaultdict
from tqdm import tqdm

import cv2
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict
from PIL import Image

import mediapy as media
import numpy as np
import torch
import tyro
from einops import repeat
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from typing_extensions import Literal, assert_never

from nerfstudio.cameras.camera_paths import get_path_from_json, get_spiral_path, get_path_from_npz
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import install_checks, vis_utils
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn

CONSOLE = Console(width=120)


def _save_gt_video(
    pipeline: Pipeline,
    output_filename: Path,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
):
    CONSOLE.print("[bold green]Creating GT video")
    outputs = pipeline.datamanager.train_dataset._dataparser_outputs
    images = [np.array(Image.open(p)) for p in outputs.image_filenames]
    # images = pipeline.datamanager.train_dataset.image_cache  # {idx: (h, w, 3)}
    # images = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]
    h, w = images[0].shape[:-1]
    output_image_dir = output_filename.parent / f'{output_filename.stem}_gt'
    output_filename = (output_filename.parent / f'{output_filename.stem}_gt').with_suffix(output_filename.suffix)
    
    if rendered_resolution_scaling_factor != 1.0:
        tgt_h, tgt_w = map(lambda x: int(round(x * rendered_resolution_scaling_factor)), [h, w])
        images = [cv2.resize(img, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA) for img in images]
    
    if output_format == "images":
        output_image_dir.mkdir(parents=True, exist_ok=True)
        for idx, gt_img in enumerate(images):
            media.write_image(output_image_dir / f"{idx:05d}.png", gt_img)
    elif output_format == "video":
        # TODO: save image for preview
        fps = len(images) / seconds
        output_filename.parent.mkdir(parents=False, exist_ok=True)
        with CONSOLE.status("[yellow]Saving GT video", spinner="bouncingBall"):
            media.write_video(output_filename, images, fps=fps)
    CONSOLE.rule("[green] :tada: :tada: :tada: Success :tada: :tada: :tada:")
    CONSOLE.print(f"[green]Saved video to {output_filename}", justify="center")


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
    save_gt: bool = False,
    save_mode: Literal["concat", "separate", "half"] = "concat",
    save_with_gt_filename: bool = False,
):
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        save_gt: save gt images at the same time
    """
    CONSOLE.print("[bold green]Creating trajectory video")
    # save gt images / videos at the same time
    if save_gt:
        _save_gt_video(pipeline, output_filename, rendered_resolution_scaling_factor,
                       seconds=seconds, output_format=output_format)
    
    images = defaultdict(list)
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)
    if save_with_gt_filename:
        gt_image_filenames = [
            Path(p).stem for p in pipeline.datamanager.train_dataset._dataparser_outputs.image_filenames
        ]
        assert len(gt_image_filenames) == cameras.size

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    output_image_dir = output_filename.parent / output_filename.stem
    output_image_dir.mkdir(parents=True, exist_ok=True)
    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
    # for camera_idx in tqdm(range(cameras.size), desc="Rendering"):
            camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx)
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                # FIXME: some of the outputs are not postprocessed for vis (e.g., normal, depth)
            render_image = {}
            for rendered_output_name in rendered_output_names:
                if rendered_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                # FIXME: outputs["rgb"] does not contain bg even when self.foreground_only is False
                output_image = outputs[rendered_output_name].cpu().numpy()
                render_image[rendered_output_name] = output_image
            
            save_frame(render_image, output_image_dir, camera_idx, save_mode=save_mode,
                        gt_filename=gt_image_filenames[camera_idx] if save_with_gt_filename else None)
            if output_format == "video":
                for _name, _img in render_image.items():
                    images[_name].append(_img)

    if output_format == "video":
        save_video(images, output_image_dir, output_filename, seconds=seconds, save_mode=save_mode)
    
    CONSOLE.rule("[green] :tada: :tada: :tada: Success :tada: :tada: :tada:")
    CONSOLE.print(f"[green]Saved video to {output_filename}", justify="center")


def save_frame(
    render_results: Dict[str, np.ndarray],
    output_image_dir: Path,
    camera_idx: int,
    save_mode: Literal["concat", "separate", "half"] = "concat",
    gt_filename: Optional[str] = None,
):
    filename = f"{camera_idx:05d}" if gt_filename is None else gt_filename
    if save_mode == "concat":
        render_image = np.concatenate(list(render_results.values()), axis=0)
        media.write_image(output_image_dir / f"{filename}.png", render_image)
        render_results["concat"] = render_image
    elif save_mode == "separate":
        for output_name, render_image in render_results.items():
            output_dir = output_image_dir / output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            if render_image.shape[-1] == 1:
                render_image = repeat(render_image, "h w 1 -> h w 3")
            media.write_image(output_dir / f"{filename}.png", render_image)
    elif save_mode == "half":
        assert len(render_results) == 2
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown save mode: {save_mode}")
    

def save_video(
    render_results: Dict[str, List[np.ndarray]],
    output_image_dir: Path,
    output_path: Path,
    seconds: float = 5.0,
    save_mode: Literal["concat", "separate", "half"] = "concat"
):
    n_frames = len(next(iter(render_results.values())))
    fps = n_frames / seconds
    if save_mode == "concat":
        images = render_results["concat"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with CONSOLE.status("[yellow]Saving video", spinner="bouncingBall"):
            media.write_video(output_path, images, fps=fps)
    elif save_mode == "separate":
        save_dir, save_fn, save_suffix = output_path.parent, output_path.stem, output_path.suffix
        save_dir.mkdir(exist_ok=True, parents=False)
        for output_name, images in render_results.items():
            _save_path = (save_dir / f'{save_fn}_{output_name}').with_suffix(save_suffix)
            with CONSOLE.status("[yellow]Saving video", spinner="bouncingBall"):
                media.write_video(_save_path, images, fps=fps)
    elif save_mode == "half":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown save mode: {save_mode}")
    
    shutil.rmtree(output_image_dir)    


@dataclass
class RenderTrajectory:
    """Load a checkpoint, render a trajectory, and save to a video file."""

    # Path to config YAML file.
    load_config: Path
    # override ckpt directory
    load_dir: Optional[Path] = None
    # Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    #  Trajectory to render.
    traj: Literal["spiral", "filename", "interpolate"] = "spiral"
    # type of trajectory file
    traj_file_type: Literal["viewer", "blender"] = "viewer"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    # Filename of the camera path to render.
    camera_path_filename: Path = Path("camera_path.json")
    # Name of the output file.
    output_path: Path = Path("renders/output.mp4")
    # How long the video should be.
    seconds: float = 5.0
    # How to save output data.
    output_format: Literal["images", "video"] = "video"
    # Specifies number of rays per chunk during eval.
    eval_num_rays_per_chunk: Optional[int] = None
    # Total number of frames to interpolate
    num_interpolate_views: int = -1
    # render foreground only
    foreground_only: bool = True
    # save gt images / videos at the same time
    save_gt: bool = False
    # num_rays_per_batch
    num_rays_per_chunk: int = 4096
    background_color: Literal["random", "last_sample", "white", "black"] = "white"
    object_bounds_extend_ratio: float = -0.05  # to avoid artifact near to fg bbox
    plane_height_ratio: Optional[float] = None  # adjust PlaneNeRF height
    save_mode: Literal["concat", "separate", "half"] = "concat"
    save_with_gt_filename: bool = False

    def main(self) -> None:
        """Main function."""
        _, pipeline, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test" if self.traj == "spiral" else "inference",
            # test_mode="inference" if self.traj == "filename" else "test",
            config_override_fn=self._config_override_fn
        )

        install_checks.check_ffmpeg_installed()

        seconds = self.seconds

        # TODO(ethan): use camera information from parsing args
        if self.traj == "spiral":
            camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0).flatten()
            # TODO(ethan): pass in the up direction of the camera
            camera_path = get_spiral_path(camera_start, steps=30, radius=0.1)
        elif self.traj == "filename":
            if self.traj_file_type == 'viewer':
                with open(self.camera_path_filename, "r", encoding="utf-8") as f:
                    camera_path = json.load(f)
                seconds = camera_path["seconds"]
                camera_path = get_path_from_json(camera_path)
            elif self.traj_file_type == 'blender':
                camera_path = np.load(self.camera_path_filename)
                seconds = camera_path["seconds"] if self.output_format == 'video' else 1
                camera_path = get_path_from_npz(camera_path)
            else:
                raise ValueError(f"Unknown traj file type: {self.traj_file_type}")
        elif self.traj == "interpolate":
            # load training data and interpolate path
            outputs = pipeline.datamanager.train_dataset._dataparser_outputs
            camera_path = vis_utils.interpolate_trajectory(cameras=outputs.cameras,
                                                           num_views=self.num_interpolate_views)
            seconds = camera_path.size / 24
        else:
            assert_never(self.traj)

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
            save_gt=self.save_gt,
            save_mode=self.save_mode,
            save_with_gt_filename=self.save_with_gt_filename
        )

    def _config_override_fn(self, config):
        if self.load_dir is not None:
            config.trainer.load_dir = self.load_dir
        if self.foreground_only:
            config.pipeline.model.render_background = False
        if self.plane_height_ratio is not None:
            config.pipeline.model.plane_height_ratio = self.plane_height_ratio
        config.pipeline.datamanager.eval_num_rays_per_batch = self.num_rays_per_chunk
        config.pipeline.model.eval_num_rays_per_chunk = self.num_rays_per_chunk
        config.pipeline.model.background_color = self.background_color
        config.pipeline.datamanager.dataparser.object_bounds_extend_ratio = self.object_bounds_extend_ratio
        return config


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[RenderTrajectory]).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderTrajectory)  # noqa
