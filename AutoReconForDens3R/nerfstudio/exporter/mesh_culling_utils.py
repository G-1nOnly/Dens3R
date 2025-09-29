
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Type
from typing_extensions import Literal
from torchtyping import TensorType
from dataclasses import dataclass, field
from rich.console import Console

import torch
import trimesh
import pymeshlab
import numpy as np
from torch.nn import functional as F

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.exporter.exporter_utils import Mesh
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.model_components.ray_samplers import Sampler, UniformSampler, DummySampler
from nerfstudio.utils.rich_utils import get_progress
from nerfstudio.configs.base_config import InstantiateConfig

CONSOLE = Console(width=120)


@dataclass
class MeshCullerConfig(InstantiateConfig):
    
    _target: Type = field(default_factory=lambda: MeshCuller)
    
    raylen_method: Literal["edge", "none"] = "edge"
    facewise_raylen: bool = False
    raylen_multiplier: float = 10.0
    """e.g., raylen = face_edge_len * raylen_multiplier"""
    use_train_ray_sampler: bool = True
    ray_sampler: Literal["uniform", "dummy"] = "uniform"
    """type of the new ray_sampler for mesh culling"""
    num_ray_samples: Optional[int] = 10  # TODO: need to specify the RaySampler as well
    """number of ray samples"""
    segmentation_binarization_thr: float = 0.5
    mesh_close_holes: bool = True
    """close holes on the mesh caused by incorrect mesh culling (bad segmentation query results)"""
    max_hole_size: int = 1000
    """maximum hole size to be closed in terms of number of edges composing the hole boundary"""
    simplify_mesh: bool = False
    """simplify mesh with decimation_quadric_edge_collapse"""


class MeshCuller:
    config: MeshCullerConfig
    
    def __init__(self, config: MeshCullerConfig):
        self.config = config
        self.ray_sampler = self._get_ray_sampler()
        
    @torch.no_grad()
    def main(self, mesh: Mesh, pipeline: Pipeline, output_path: Path):
        """Cull mesh by rendering the segmentation of each face."""
        device = pipeline.device

        vertices = mesh.vertices.to(device)  # (nv, 3)
        faces = mesh.faces.to(device)  # (nf, 3)
        face_normals = mesh.face_normals.to(device)  # (nf, 3)
        ray_bundle = self._build_ray_bundle(vertices, faces, face_normals)
        
        outputs = pipeline.model.get_outputs_for_mesh_culling(ray_bundle, ray_sampler=self.ray_sampler)
        
        # TODO: support only cull faces near to the ground plane
        
        # TODO: if a faces is not visible from any viewpoint (or only visible from a very few viewpoints)
        # we could just keep it, since its feature is not learned by feature field.
        
        fg_seg_prob = outputs["fg_seg_prob"]
        fg_faces_mask = (fg_seg_prob > self.config.segmentation_binarization_thr)[..., 0]
        CONSOLE.print(f"{len(fg_faces_mask) - fg_faces_mask.sum()}/{len(fg_faces_mask)} faces get culled!")
        faces = faces[fg_faces_mask]  # (nf', 3)
        face_normals = face_normals[fg_faces_mask]  # (nf', 3)
        
        culled_mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(),
                                      faces=faces.cpu().numpy(),
                                      face_normals=face_normals.cpu().numpy())
        culled_mesh.export(output_path)
        CONSOLE.print(f'Culled mesh saved: {output_path}')
        
        if self.config.mesh_close_holes:
            hole_closed_path = output_path.parent / (output_path.stem + "_hole-closed" + output_path.suffix)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(output_path))
            CONSOLE.print('Running mesh hole closing...')
            ms.meshing_close_holes(maxholesize=self.config.max_hole_size)
            ms.save_current_mesh(str(hole_closed_path), save_face_color=False)
            CONSOLE.print(f'Hole-closed mesh saved: {hole_closed_path}')
            output_path = hole_closed_path
            
        if self.config.simplify_mesh:
            simplified_path = output_path.parent / (output_path.stem + "_simplified" + output_path.suffix)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(output_path))

            CONSOLE.print('Running remashing...')
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=2000000)
            ms.save_current_mesh(str(simplified_path), save_face_color=False)
            CONSOLE.print(f'Simplified mesh saved: {simplified_path}')
            output_path = simplified_path
        return output_path

    def _get_ray_sampler(self):
        if self.config.use_train_ray_sampler:
            return None
        
        ray_sampler_type = self.config.ray_sampler
        num_samples = self.config.num_ray_samples
        if ray_sampler_type == "uniform":
            _log_str = f"UniformSampler(num_samples={num_samples})"
            sampler = UniformSampler(num_samples=num_samples)
        elif ray_sampler_type == "dummy":
            _log_str = "DummySampler"
            sampler = DummySampler()
        else:
            raise ValueError(f"Unknown RaySampler type: {ray_sampler_type}")
        CONSOLE.print(f"Replace the original sampler with {_log_str}.")
        return sampler
    
    def _build_ray_bundle(self, vertices, faces, face_normals) -> RayBundle:
        face_vertices = vertices[faces]  # (nf, 3, 3)
        raylen = self._get_ray_length(vertices, faces)  # (nf, )
        origins = face_vertices.mean(dim=1)  # (nf, 3)
        directions = F.normalize(face_normals, dim=-1)
        if self.config.ray_sampler != "dummy":
            origins = origins - 0.5 * raylen * directions
        ray_bundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=torch.ones_like(origins[..., 0:1]),
            camera_indices=torch.zeros_like(origins[..., 0:1]),
            nears=torch.zeros_like(origins[..., 0:1]),
            fars=torch.ones_like(origins[..., 0:1]) * raylen
        )
        return ray_bundle
    
    def _get_ray_length(self, vertices, faces):
        raylen_method = self.config.raylen_method
        facewise_raylen = self.config.facewise_raylen
        raylen_multiplier = self.config.raylen_multiplier
        
        if raylen_method == "edge":
            face_vertices = vertices[faces]
            # compute the length of the rays we want to render
            # we make a reasonable approximation by using the mean length of one edge per face
            raylen = raylen_multiplier * torch.norm(face_vertices[:, 1, :] - face_vertices[:, 0, :], dim=-1)  # (nf, )
            if not facewise_raylen:
                raylen = torch.full_like(raylen, float(torch.mean(raylen)))
        else:
            raise ValueError(f"Ray length method {raylen_method} not supported.")
        return raylen[..., None]
