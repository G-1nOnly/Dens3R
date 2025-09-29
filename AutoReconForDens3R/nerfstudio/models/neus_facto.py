# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of VolSDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Type, Tuple, Dict
from typing_extensions import Literal
import numpy as np

import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import interlevel_loss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.utils import colormaps


@dataclass
class NeuSFactoModelConfig(NeuSModelConfig):
    """UniSurf Model Config"""

    _target: Type = field(default_factory=lambda: NeuSFactoModel)
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for the proposal network."""
    num_neus_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000  # FIXME: not used
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 64},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256},
        ]
    )
    """Arguments for the proposal density fields."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    proposal_net_spatial_normalization_region: Literal["full", "fg"] = "full"
    """if "full", normalize [-2, 2] into [0, 1], otherwise, normalize [-1, 1] into [0, 1]."""
    proposal_use_uniform_sampler: bool = True
    """whether to use UniformSampler as the initial sampler for the proposal network.
    It's better to use UniformSampler if the region for the proposal network to model
    is bounded and small (e.g., inside a object bbox). UniformLinDispPiecewiseSampler is more
    appropriate, if the proposal network needs to model a big or unbounded region.
    """
    proposal_net_use_separate_contraction: bool = False
    """use a separate contraction for the proposal network (modeling both fg & bg regions)."""
    proposal_net_contraction_scale_factor: float = 1.0
    """scale_factor for the separate SceneContraction of the proposal network."""
    num_proposal_samples_per_ray_for_eikonal_loss: int = 0


class NeuSFactoModel(NeuSModel):
    """NeuS facto model

    Args:
        config: NeuS configuration to instantiate model
    """

    config: NeuSFactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        
        # Build the proposal network(s)
        if self.config.proposal_net_use_separate_contraction:
            proposal_contraction = SceneContraction(
                order=float("inf"), scale_factor=self.config.proposal_net_contraction_scale_factor
            )
        else:
            proposal_contraction = self.scene_contraction
        
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb, spatial_distortion=proposal_contraction, **prop_net_args,
                spatial_normalization_region=self.config.proposal_net_spatial_normalization_region,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[
                    min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb, spatial_distortion=proposal_contraction, **prop_net_args,
                    spatial_normalization_region=self.config.proposal_net_spatial_normalization_region,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # proposal network scheduler
        # TODO: previously trained NeuSFacto models use update_scheduler=-1 -> might lead to performance change
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )
        
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_neus_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            use_uniform_sampler=self.config.proposal_use_uniform_sampler,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

        return callbacks

    def sample_and_forward_field(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )

        field_outputs = self.field(ray_samples, return_alphas=True)
        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs[FieldHeadNames.ALPHA]
        )
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

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training:
            self._compute_interlevel_loss(outputs, loss_dict)
            self._compute_proposal_eikonal_loss(outputs, loss_dict)

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            # TODO: set depth[accumulation < 0.1] = depth[accumulation >= 0.1].min()
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],  # NOTE: this would remove bg depths
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    def _compute_interlevel_loss(self, outputs: Dict[str, torch.Tensor], loss_dict: Dict[str, torch.Tensor]):
        weights_list, ray_samples_list = self._get_interlevel_loss_inputs(outputs)
        interlevel_loss_val = interlevel_loss(weights_list, ray_samples_list)
        loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss_val
    
    def _get_interlevel_loss_inputs(self, outputs):
        weights_list = outputs["weights_list"]  # (n_rays, n_samples, 1)
        ray_samples_list = outputs["ray_samples_list"]  # (n_rays, n_samples)
        
        if not self.handle_no_intersect_rays:
            return weights_list, ray_samples_list
        
        # ignore rays not intersecting the fg aabb
        no_intersect_mask = outputs["no_intersect_mask"][..., 0]  # (n_rays, )
        weights_list = [weights[~no_intersect_mask] for weights in weights_list]
        ray_samples_list = [ray_samples[~no_intersect_mask] for ray_samples in ray_samples_list]
        
        return weights_list, ray_samples_list

    def _compute_proposal_eikonal_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        loss_dict: Dict[str, torch.Tensor]
    ):
        """sample additional ray samples from the proposal samples and compute the eikonal loss.
        this aims to regularize the internal geometry of the reconstruction.
        """
        n_eikonal_samples = self.config.num_proposal_samples_per_ray_for_eikonal_loss
        if n_eikonal_samples <= 0:
            return
        
        _, ray_samples_list = self._get_interlevel_loss_inputs(outputs)  # TODO: called twice per iter, should cache results
        ray_samples_list = [s.frustums.get_start_positions() for s in ray_samples_list[:-1]]
        # TODO: maybe it's better to just sample from the uniform samples?
        ray_samples = torch.cat(ray_samples_list, dim=1)  # (n_rays, n_samples, 3)
        n_rays, n_samples = ray_samples.shape[:2]
        if n_eikonal_samples < n_samples:  # sample with replacement for speed
            _inds = torch.randint(n_samples, (n_rays, n_eikonal_samples), device=ray_samples.device)
            ray_samples = ray_samples.gather(1, _inds[..., None].expand(-1, -1, 3))
        eik_grads = self.field.gradient(ray_samples.view(-1, 3))  # (*, 3)
        loss_dict["proposal_eikonal_loss"] = ((eik_grads.norm(2, dim=-1) - 1) ** 2).mean() * self.config.eikonal_loss_mult
