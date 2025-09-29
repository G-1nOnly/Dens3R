#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Literal, Optional, Tuple

import igl
import numpy as np
import open3d as o3d
import pymeshlab
import torch
import tyro
from rich.console import Console
from scipy import cluster

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nerfstudio.exporter import texture_utils
from nerfstudio.exporter.exporter_utils import get_mesh_from_filename
from nerfstudio.exporter.mesh_culling_utils import MeshCuller, MeshCullerConfig
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.marching_cubes import get_surface_occupancy, get_surface_sliding

CONSOLE = Console(width=120)


@dataclass
class ExtractMesh:
    """Load a checkpoint, run marching cubes, extract mesh, and save it to a ply file."""

    load_config: Path
    """Path to config YAML file."""
    resolution: int = 1024
    """Marching cube resolution."""
    output_path: Path = Path("output.ply")
    """Name of the output file."""
    simplify_mesh: bool = False
    """Whether to simplify the mesh."""
    is_occupancy: bool = False
    """extract the mesh using occupancy field (unisurf) or SDF, default sdf"""
    chunk_size: int = 100000
    """sdf query chunk size"""
    store_float16: bool = False
    """store intermediate results in float16"""

    use_train_scene_box: bool = False
    """read DataParser's SceneBox as bonding box"""
    train_scene_box_extend_ratio: float = 0.0
    """override dataparser config"""
    train_scene_box_extend_ground_plane_only: bool = False
    """override dataparser config"""
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    """Minimum of the bounding box."""
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Maximum of the bounding box."""

    seg_aware_sdf: bool = False
    """Rectify raw sdf values based on the corresponding segmentation results"""
    mesh_culling_enabled: bool = False
    """joint segementation rendering and mesh culling"""
    mesh_culling: MeshCullerConfig = field(default_factory=MeshCullerConfig)
    """config for mesh culling"""
    remove_non_maximum_connected_components: bool = False
    """delete non-maximum connected components in the produces mesh"""

    remove_internal_geometry: Optional[Literal["meshlab", "igl"]] = None
    """remove internal mesh geometry with ambient occlusion"""

    close_holes: bool = False
    max_hole_size: int = 1000
    simplify_mesh_final: bool = False
    target_face_num: int = 1000000

    extract_texture: bool = False
    """extract texture with NeRF"""
    target_num_faces: Optional[int] = None
    """Target number of faces for the mesh to texture."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""

    # override original config fields
    load_dir: Optional[Path] = None
    downsample_ptcd: int = 1000
    feature_seg_knn: int = 10
    feature_seg_dist_weighted_avg: bool = False
    feature_seg_separate_bg_plane: bool = False

    def main(self) -> None:
        """Main function."""
        assert str(self.output_path)[-4:] == ".ply"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = self.output_path

        _, pipeline, _ = eval_setup(self.load_config, cache_images=False, config_override_fn=self._config_override_fn)
        if self.use_train_scene_box:
            aabb = pipeline.datamanager.train_dataset.scene_box.aabb
            self.bounding_box_min = aabb[0].cpu().numpy()
            self.bounding_box_max = aabb[1].cpu().numpy()
            _bbox_str = str(np.stack([self.bounding_box_min, self.bounding_box_max], 0).round(2))
            CONSOLE.print(f"Using training SceneBox for mesh extraction: {_bbox_str}")

        CONSOLE.print("Extract mesh with marching cubes and may take a while")

        if self.is_occupancy:
            assert not self.seg_aware_sdf
            # for unisurf
            get_surface_occupancy(
                occupancy_fn=lambda x: torch.sigmoid(
                    10 * pipeline.model.field.forward_geonetwork(x)[:, 0].contiguous()
                ),
                resolution=self.resolution,
                bounding_box_min=self.bounding_box_min,
                bounding_box_max=self.bounding_box_max,
                level=0.5,
                device=pipeline.model.device,
                output_path=self.output_path,
                chunk_size=self.chunk_size,
            )
        else:
            # assert self.resolution % 256 == 0
            # for sdf we can multi-scale extraction.
            get_surface_sliding(
                sdf=partial(self.extract_sdf, pipeline),
                resolution=self.resolution,
                bounding_box_min=self.bounding_box_min,
                bounding_box_max=self.bounding_box_max,
                coarse_mask=pipeline.model.scene_box.coarse_binary_gird,
                output_path=self.output_path,
                simplify_mesh=self.simplify_mesh,
                chunk_size=self.chunk_size,
                store_float16=self.store_float16,
            )

        if self.mesh_culling_enabled:
            mesh_culler: MeshCuller = self.mesh_culling.setup()
            mesh = get_mesh_from_filename(str(self.output_path), target_num_faces=None)
            output_path = self.output_path.parent / (self.output_path.stem + "_culled" + self.output_path.suffix)
            output_path = mesh_culler.main(mesh, pipeline, output_path)

        output_path = self._remove_internal_geometry(output_path)
        output_path = self._remove_isolated_components(output_path)
        output_path = self._fix_mesh(output_path)

        if self.extract_texture:
            textured_mesh_dir = self._extract_texture(pipeline, output_path)

    def extract_sdf(self, pipeline, x):
        if self.seg_aware_sdf:
            sdf_val = pipeline.model.seg_aware_sdf(x)
        else:
            sdf_val = pipeline.model.field.forward_geonetwork(x)
        return sdf_val[:, 0].contiguous()

    def _remove_internal_geometry(self, output_path: Path) -> Path:
        if self.remove_internal_geometry is None:
            return output_path
        elif self.remove_internal_geometry == "meshlab":
            return self._remove_internal_geometry_meshlab(output_path)
        elif self.remove_internal_geometry == "igl":
            return self._remove_internal_geometry_igl(output_path)
        else:
            raise ValueError(f"Unknown method: {self.remove_internal_geometry}")

    def _remove_internal_geometry_igl(
        self, output_path: Path, n_samples: int = 512, min_visible_ratio: float = 0.01
    ) -> Path:
        new_output_path = output_path.parent / (output_path.stem + "_internal-removed" + output_path.suffix)
        v, f = igl.read_triangle_mesh(str(output_path))
        n = igl.per_vertex_normals(v, f)
        ao = igl.ambient_occlusion(v, f, v, n, n_samples)
        face_ao = np.sum(ao[f], axis=1) / 3.0  # 1.0 if fully occluded, 0.0 if fully visible
        face_vis_ratio = 1.0 - face_ao
        nv, nf = v, f[face_vis_ratio >= min_visible_ratio]
        n_faces_removed = len(f) - len(nf)
        CONSOLE.print(
            f"Mesh internal geometry removed w. igl (AO, #faces_removed={n_faces_removed}) and saved: {new_output_path}"
        )
        igl.write_triangle_mesh(str(new_output_path), nv, nf, force_ascii=False)

    def _remove_internal_geometry_meshlab(self, output_path: Path) -> Path:
        new_output_path = output_path.parent / (output_path.stem + "_internal-removed" + output_path.suffix)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(output_path))
        ms.compute_scalar_ambient_occlusion(
            occmode="per-Vertex", dirbias=0, reqviews=1024, usegpu=True, depthtexsize=512
        )
        ms.compute_selection_by_condition_per_vertex(condselect="(q < 0.01)")
        ms.meshing_remove_selected_vertices()
        ms.meshing_remove_null_faces()
        ms.save_current_mesh(str(new_output_path), save_face_color=False)
        CONSOLE.print(f"Mesh internal geometry removed w. meshlab (AO) and saved: {new_output_path}")
        return new_output_path

    def _remove_isolated_components(self, output_path: Path) -> Path:
        """remove internal isolated geometries caused by hash encoding and insufficent regularization of
        internal regions.
        """
        if not self.remove_non_maximum_connected_components:
            return output_path

        CONSOLE.print("Extracting maximumally-connected components...")
        isolation_removed_path = output_path.parent / (output_path.stem + "_max-component" + output_path.suffix)
        # ms = pymeshlab.MeshSet()
        # ms.load_new_mesh(str(output_path))
        # CONSOLE.print('Removing non-maximumally-connected components...')
        # ms = ms.generate_splitting_by_connected_components()[0]  # This is too slow
        mesh = o3d.io.read_triangle_mesh(str(output_path))
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)  # (n_triangles, )
        cluster_n_triangles = np.asarray(cluster_n_triangles)  # (n_clusters, )
        triangles_to_remove = triangle_clusters != cluster_n_triangles.argmax()
        mesh.remove_triangles_by_mask(triangles_to_remove)
        o3d.io.write_triangle_mesh(str(isolation_removed_path), mesh)
        CONSOLE.print(f"Maximally-connected mesh saved: {isolation_removed_path}")
        return isolation_removed_path

    def _fix_mesh(self, output_path: Path) -> Path:
        if not (self.close_holes or self.simplify_mesh):
            return output_path

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(output_path))

        # close holes
        if self.close_holes:
            output_path = output_path.parent / (output_path.stem + "_hole-closed" + output_path.suffix)
            ms.meshing_close_holes(maxholesize=self.max_hole_size)
            CONSOLE.print(f"Mesh Hole-closed w/ max hole size = {self.max_hole_size}")

        # simplify mesh
        if self.simplify_mesh_final:
            output_path = output_path.parent / (output_path.stem + "_simplified" + output_path.suffix)
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=self.target_face_num)
            CONSOLE.print(f"Remeshing donw w/ targetfacenum = {self.target_face_num}")

        ms.save_current_mesh(str(output_path), save_face_color=False)
        CONSOLE.print(f"Fixed mesh saved: {output_path}")
        return output_path

    def _extract_texture(self, pipeline, mesh_path: Path) -> Path:
        mesh = get_mesh_from_filename(str(mesh_path), target_num_faces=None)
        CONSOLE.print("Texturing mesh with NeRF")

        output_dir = mesh_path.parent / (mesh_path.stem + "_textured")
        texture_utils.export_textured_mesh(
            mesh,
            pipeline,
            output_dir,
            px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
            unwrap_method=self.unwrap_method,
            num_pixels_per_side=self.num_pixels_per_side,
        )
        return output_dir

    def _config_override_fn(self, config):  # TODO: make optional
        """override config fields manually"""
        if self.load_dir is not None:
            config.trainer.load_dir = self.load_dir

        if self.use_train_scene_box:
            config.pipeline.datamanager.dataparser.object_bounds_extend_ratio = self.train_scene_box_extend_ratio
            config.pipeline.datamanager.dataparser.extend_ground_plane_only = (
                self.train_scene_box_extend_ground_plane_only
            )

        if hasattr(config.pipeline.model, "feature_seg_field"):
            config.pipeline.model.feature_seg_field.knn = self.feature_seg_knn
            config.pipeline.model.feature_seg_field.distance_weighted_average = self.feature_seg_dist_weighted_avg
            config.pipeline.model.feature_seg_field.separate_bg_plane = self.feature_seg_separate_bg_plane

        config.pipeline.datamanager.dataparser.downsample_ptcd = self.downsample_ptcd
        return config


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[ExtractMesh]).main()


if __name__ == "__main__":
    entrypoint()


# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ExtractMesh)  # noqa
