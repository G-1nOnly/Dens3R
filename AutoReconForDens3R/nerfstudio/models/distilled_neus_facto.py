"""
Two-stages training of NeuSFacto: 1. use NeRF in both fg and bg; 2. replace NeRF with NeuSFacto in fg.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Type, Tuple, Dict, Any
from typing_extensions import Literal
import numpy as np

import torch
from torch import nn

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.models.neus_facto import NeuSFactoModelConfig, NeuSFactoModel
from nerfstudio.model_components.losses import interlevel_loss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.utils import colormaps


@dataclass
class DistilledNeuSFactoModelConfig(NeuSFactoModelConfig):
    """Distilled NeuSFactoModel config"""
    _target: Type = field(default_factory=lambda: DistilledNeuSFactoModel)
    nerfacto_field_enabled: bool = True
    """if True, use nerfacto field in fg; otherwise, use sdf_field in fg"""
    nerfacto_hash_grid_num_levels: int = 16
    nerfacto_hash_grid_max_res: int = 2048
    nerfacto_hash_grid_log2_hashmap_size: int = 19
    nerfacto_predict_normals: bool = False
    nerfacto_use_appearance_embedding: bool = False
    nerfacto_spatial_normalization_region: Literal["full", "fg"] = "fg",
    single_field_fg_bg: bool = False
    """if True, use a single nerfactor field for fg and bg"""
    ignore_pre_intersection_points: bool = True
    """ignore samples between the camera origin and the first ray-aabb intersection point in background modeling."""
    
    
class DistilledNeuSFactoModel(NeuSFactoModel):
    
    config: DistilledNeuSFactoModelConfig
    
    def populate_modules(self):
        super().populate_modules()
        self.aabb = nn.Parameter(self.scene_box.aabb, requires_grad=False)
        
        # init nerfacto field (override the original sdf_field)
        if self.config.nerfacto_field_enabled:
            self.field = TCNNNerfactoField(
                self.scene_box.aabb,
                num_levels=self.config.nerfacto_hash_grid_num_levels,
                max_res=self.config.nerfacto_hash_grid_max_res,
                log2_hashmap_size=self.config.nerfacto_hash_grid_log2_hashmap_size,
                spatial_distortion=self.scene_contraction,
                num_images=self.num_train_data,
                use_pred_normals=self.config.nerfacto_predict_normals,
                use_appearance_embedding=self.config.nerfacto_use_appearance_embedding,
                use_average_appearance_embedding=self.config.use_average_appearance_embedding,
                spatial_normalization_region=self.config.nerfacto_spatial_normalization_region,
            )
        
        if self.config.single_field_fg_bg and self.config.background_model != "none":  # override the original bg field
            self.field_background = self.field
        
        assert isinstance(self.collider, NearFarCollider), "DistilledNeuSFactoModel only supports NearFarCollider"
        
    def sample_and_forward_field(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )
        # split ray_samples into fg and bg samples and update ray_samples.metadata
        self._mask_ray_samples_with_aabb(ray_bundle, ray_samples)
        # TODO: masked inference in foreground and background fields
        weights, transmittance, field_outputs = self.forward_field(ray_samples)
        bg_transmittance = transmittance[:, -1, :]
        
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)
        
        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "bg_transmittance": bg_transmittance,
            "weights_list": weights_list,
            "ray_samples_list": ray_samples_list,
        }
        return samples_and_field_outputs
        
    def sample_and_forward_field_bg(
        self, ray_bundle: RayBundle,
        samples_and_field_outputs_fg: Dict[str, Any],
        outputs: Dict[str, Any],
    ):
        ray_samples = samples_and_field_outputs_fg["ray_samples"]  # fg & bg share the same set of ray samples
        bg_transmittance = samples_and_field_outputs_fg["bg_transmittance"]
        ray_samples_bg_mask = ray_samples.metadata["bg_mask"]
        
        # forward the bg field and mask out weights of invalid positions
        field_outputs_bg = self.field_background(ray_samples)
        densities = field_outputs_bg[FieldHeadNames.DENSITY]
        densities = densities * ray_samples_bg_mask.to(densities.dtype)
        # densities = torch.where(ray_samples_bg_mask, densities, torch.zeros_like(densities))
        weights_bg = ray_samples.get_weights(densities) * bg_transmittance[:, None]
        weights_fg = samples_and_field_outputs_fg["weights_list"][-1]
        weights_fg_bg = torch.where(ray_samples_bg_mask, weights_bg, weights_fg)
        samples_and_field_outputs_fg["weights_list"][-1] = weights_fg_bg
        field_outputs_bg.update({FieldHeadNames.DENSITY: densities})
        
        samples_and_field_outputs_bg = {
            "ray_samples": ray_samples,
            "weights": weights_bg,
            **field_outputs_bg
        }
        return samples_and_field_outputs_bg
    
    def forward_field(self, ray_samples):
        """Forward the fg field and mask out weights of points outside the fg aabb.
        """
        if self.config.nerfacto_field_enabled:
            return self._forward_field_nerfacto(ray_samples)
        else:
            return self._forward_field_sdf(ray_samples)
    
    def _forward_field_sdf(self, ray_samples):
        field_outputs = self.field(ray_samples, return_alphas=True)
        alphas = field_outputs[FieldHeadNames.ALPHA]
        if ray_samples.metadata is not None and "fg_mask" in ray_samples.metadata:
            alphas = alphas * ray_samples.metadata["fg_mask"].to(alphas.dtype)
        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(alphas)
        field_outputs.update({FieldHeadNames.ALPHA: alphas})
        return weights, transmittance, field_outputs
    
    def _forward_field_nerfacto(self, ray_samples):
        field_outputs = self.field(ray_samples, compute_normals=self.config.nerfacto_predict_normals)
        densities = field_outputs[FieldHeadNames.DENSITY]
        if ray_samples.metadata is not None and "fg_mask" in ray_samples.metadata:
            densities = densities * ray_samples.metadata["fg_mask"].to(densities.dtype)
        weights, transmittance = ray_samples.get_weights_and_transmittance(densities)
        field_outputs.update({FieldHeadNames.DENSITY: densities})
        return weights, transmittance, field_outputs
        
    def _mask_ray_samples_with_aabb(self, ray_bundle: RayBundle, ray_samples: RaySamples) -> None:
        """Compute per ray_sample fg masks and bg masks"""
        if self.config.background_model == "none":
            return
        
        aabb = self.aabb  # (2, 3)
        
        positions = ray_samples.frustums.get_positions()  # (n_rays, n_pts, 3)
        in_aabb_mask = ((positions >= aabb[[0], None]) & (positions <= aabb[[1], None])).all(-1, keepdim=True)
        out_aabb_mask = ~in_aabb_mask
        
        if self.config.ignore_pre_intersection_points:
            # TODO: support the modeling of regions b/w camera near & the 1st aabb intersection (currently ignored)
            nears = self._compute_aabb_intersections(ray_bundle)[:, None, None]  # (n_rays, 1, 1)
            ray_dists = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
            out_ignore_mask = ray_dists <= nears  # (n_rays, n_pts, 1)
            out_aabb_mask[out_ignore_mask] = False
        else:
            raise NotImplementedError  # three part composition
            # # additionally consider pre-intersection points
            # nears = self._compute_aabb_intersections(ray_bundle)[:, None, None]  # (n_rays, 1, 1)
            # ray_dists = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
            # in_additional_mask = ray_dists <= nears  # (n_rays, n_pts, 1)
            # in_aabb_mask[in_additional_mask] = True
            # out_aabb_mask = ~in_aabb_mask
        
        if ray_samples.metadata is None:
            ray_samples.metadata = {}
        ray_samples.metadata["fg_mask"] = in_aabb_mask
        ray_samples.metadata["bg_mask"] = out_aabb_mask
    
    def _compute_aabb_intersections(self, ray_bundle: RayBundle):
        aabb = self.aabb
        rays_o, rays_d = ray_bundle.origins, ray_bundle.directions
        dir_fraction = 1.0 / (rays_d + 1e-6)
        
        # x
        t1 = (aabb[0, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
        t2 = (aabb[1, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
        # y
        t3 = (aabb[0, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
        t4 = (aabb[1, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
        # z
        t5 = (aabb[0, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]
        t6 = (aabb[1, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]
        
        tx_min, tx_max = torch.minimum(t1, t2), torch.maximum(t1, t2)
        ty_min, ty_max = torch.minimum(t3, t4), torch.maximum(t3, t4)
        tz_min, tz_max = torch.minimum(t5, t6), torch.maximum(t5, t6)
        
        nears = torch.max(torch.cat([tx_min, ty_min, tz_min], dim=1), dim=1).values
        fars = torch.min(torch.cat([tx_max, ty_max, tz_max], dim=1), dim=1).values
        nears[nears > fars] = -1.0  # no intersection with the first plane -> nears < 0
        return nears
