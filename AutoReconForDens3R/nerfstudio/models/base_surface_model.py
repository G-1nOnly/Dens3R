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
Implementation of Base surface model.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from rich.console import Console
from torch.nn import ModuleList, Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import (
    ForegroundAwareSceneContraction,
    SceneContraction,
)
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    MultiViewLoss,
    ScaleAndShiftInvariantLoss,
    SensorDepthLoss,
    compute_scale_and_shift,
    interlevel_loss,
    monosdf_normal_loss,
)
from nerfstudio.model_components.patch_warping import PatchWarping
from nerfstudio.model_components.ray_samplers import (
    LinearDisparitySampler,
    ProposalNetworkSampler,
    UniformLinDispPiecewiseSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    SemanticRenderer,
)
from nerfstudio.model_components.scene_colliders import (
    AABBBoxCollider,
    AABBBoxNearFarCollider,
    NearFarCollider,
    SphereCollider,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, profiler, scheduler, writer
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.pointclouds import BasicPointClouds

CONSOLE = Console(width=120)


def map_range_val(input_val, input_start, input_end, output_start, output_end, type='linear', factor=1.):
    input_clamped=max(input_start, min(input_end, input_val))
    input_relative = (input_clamped - input_start) / (input_end - input_start)

    if type == 'linear':
        return output_start + (output_end - output_start) * input_relative
    elif type == 'exp':
        return output_start + (output_end - output_start) * (np.exp(factor*input_relative) - 1) / (np.exp(factor) - 1)
    elif type == 'log':
        return output_start + (output_end - output_start) * np.log1p(factor*input_relative) / np.log(1+factor) # log base 2
    else:
        raise NotImplementedError

@dataclass
class SurfaceModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: SurfaceModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 4.0
    """How far along the ray to stop sampling."""
    far_plane_bg: float = 1000.0
    """How far along the ray to stop sampling of the background model."""
    background_color: Literal["random", "last_sample", "white", "black"] = "black"
    """Whether to randomize the background color."""
    # fg_use_appearance_embedding: bool = False  # NOTE: directly set "sdf_field.use_appearance_embedding" instead
    # """Whether to use per-image appearance embedding in the fg model."""
    bg_use_appearance_embedding: bool = True
    """Whether to use per-image appearance embedding in the bg model."""
    bg_sampler_type: Literal["uniform_lin_disp", "lin_disp", "proposal_network", "none"] = "lin_disp"
    use_proposal_weight_anneal_bg: bool = True
    use_average_appearance_embedding: bool = (
        True  # FIXME: False use zero appearance embedding during inference, which leads to bad bg rendering.
    )
    """Whether to use average appearance embedding or zeros for inference."""
    eikonal_loss_mult: float = 0.1
    """Monocular normal consistency loss multiplier."""
    fg_mask_loss_mult: float = 0.01
    """Foreground mask loss multiplier."""
    mono_normal_loss_mult: float = 0.002
    """Monocular normal consistency loss multiplier."""
    mono_depth_loss_mult: float = 0.0
    """Monocular depth consistency loss multiplier."""
    patch_warp_loss_mult: float = 0.0
    """Multi-view consistency warping loss multiplier."""
    patch_size: int = 11
    """Multi-view consistency warping loss patch size."""
    patch_warp_angle_thres: float = 0.3
    """Threshold for valid homograph of multi-view consistency warping loss"""
    min_patch_variance: float = 0.01
    """Threshold for minimal patch variance"""
    topk: int = 4
    """Number of minimal patch consistency selected for training"""
    sensor_depth_truncation: float = 0.015
    """Sensor depth trunction, default value is 0.015 which means 5cm with a rough scale value 0.3 (0.015 = 0.05 * 0.3)"""
    sensor_depth_l1_loss_mult: float = 0.0
    """Sensor depth L1 loss multiplier."""
    sensor_depth_freespace_loss_mult: float = 0.0
    """Sensor depth free space loss multiplier."""
    sensor_depth_sdf_loss_mult: float = 0.0
    """Sensor depth sdf loss multiplier."""
    sparse_points_sdf_loss_mult: float = 0.0
    """sparse point sdf loss multiplier"""
    sdf_field: SDFFieldConfig = SDFFieldConfig()
    """Config for SDF Field"""
    background_model: Literal["grid", "mlp", "none"] = "mlp"
    """background models"""
    render_background: bool = True
    """optionally disable bg rendering during testing"""
    bg_hash_grid_num_levels: int = 16
    """num_levels for background hash grid"""
    num_samples_outside: int = 32
    """Number of samples outside the bounding sphere for backgound"""
    num_bg_proposal_samples_per_ray: Tuple[int, int] = (128, 64)
    handle_no_intersection_rays: bool = True
    """whether to separately handle rays not intersecting the scene, usefull only when using AABBBoxNearFarCollider"""
    handle_bottom_intersection_rays: bool = False
    """if a ray intersects the bottom plane of the fg aabb, ignore the querying of the bg model"""
    periodic_tvl_mult: float = 0.0
    """Total variational loss mutliplier"""
    overwrite_near_far_plane: bool = False
    """whether to use near and far collider from command line"""
    scene_contraction_order: Literal["none", "inf"] = "inf"
    """norm of the scene contraction, use l2-norm if none"""
    scene_contraction_scale_factor: float = 1.0
    """Scale positions by this factor before applying contraction.
    This is useful when the [-1, 1] region (no contraction) is too small and only covers a small portion of the scene.
    """
    use_fg_aware_scene_contraction: bool = False
    """Use ForegroundAwareSceneContraction, which makes better use of the bg feature grid"""
    fg_aware_scene_contraction_alpha: float = 5.0
    """map 1 < ||x|| <= alpha region to [[-1, -1/alpha], [1/alpha, 1]], bigger alpha leaves more space linearly uncontracted"""
    curvature_loss_mult: float = 0.0
    """curvature loss in HashSDF"""
    curvature_loss_n_iters: Optional[int] = None
    """number of iterations to apply curvature loss"""
    ptcd_reg_fg_mult: float = 0.0
    """regularization with fg pointclouds"""
    ptcd_reg_fg_n_pts_per_iter: int = 5000
    ptcd_reg_plane_mult: float = 0.0
    """regularization with plane pointclouds"""
    ptcd_reg_plane_n_pts_per_iter: int = 5000
    ptcd_reg_n_iters: Optional[int] = None  # if None, apply ptcd_reg for the entire training
    ptcd_reg_weight_anneal_fn: Literal["cosine", "exponential", "constant"] = "cosine"
    mask_beta_prior_mult: float = 0.0
    """beta prior regularization for the fg accumulation"""
    mask_beta_prior_n_iters: Optional[int] = None
    mask_beta_prior_anneal_fn: Literal["sine", "cosine", "exponential", "constant"] = "sine"
    depth_renderer_mode: Literal["expected", "median"] = "expected"
    mask_depth_with_acc: bool = False
    
    mono_normal_loss_warmup_steps: int = 5000
    mono_normal_loss_steps: int = 20000
    """ normal warmup steps """

    # init_normal_cos_thresh: float = 0.1 
    normal_loss_final_scale: float = 5
    # normal_decay_weight: float = 0.99995
    """ adaptive normal loss """

class SurfaceModel(Model):
    """Base surface model

    Args:
        config: Base surface model configuration to instantiate model
    """

    config: SurfaceModelConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handle_no_intersect_rays = self.config.handle_no_intersection_rays and isinstance(
            self.collider, (AABBBoxNearFarCollider, SphereCollider)
        )
        self.handle_bottom_intersect_rays = self.config.handle_bottom_intersection_rays and isinstance(
            self.collider, AABBBoxNearFarCollider
        )
        self._step = 0
        self.ptcd_reg_anneal_fn = self._build_anneal_fn(
            self.config.ptcd_reg_weight_anneal_fn, self.config.ptcd_reg_n_iters
        )
        if self.config.ptcd_reg_plane_mult > 0 and self.config.ptcd_reg_fg_mult == 0:
            CONSOLE.print(
                "[bold red]Using plane-ptcd-reg w/o fg-ptcd-reg, this could degenerate fg reconstruction!"
            )  # due to imbalanced pt sampling
        # TODO: build separate loss classes intergrating loss computation & weight scheduling

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self._build_scene_contraction()

        self.field = self.config.sdf_field.setup(
            aabb=self.scene_box.aabb,
            spatial_distortion=self.scene_contraction,
            num_images=self.num_train_data,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            compute_curvature_loss=self.config.curvature_loss_mult > 0,
        )

        self._build_collider()
        self._build_bg_model_and_sampler()
        self._build_renderers()

        # patch warping
        self.patch_warping = PatchWarping(
            patch_size=self.config.patch_size, valid_angle_thres=self.config.patch_warp_angle_thres
        )

        # ptcd regularization
        if self.config.ptcd_reg_fg_mult > 0 or self.config.ptcd_reg_plane_mult > 0:
            self.ptcd_data: BasicPointClouds = self.kwargs["metadata"]["ptcd_data"]

        # losses
        self.rgb_loss = L1Loss()
        self.eikonal_loss = MSELoss()
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        self.patch_loss = MultiViewLoss(
            patch_size=self.config.patch_size, topk=self.config.topk, min_patch_variance=self.config.min_patch_variance
        )
        self.sensor_depth_loss = SensorDepthLoss(truncation=self.config.sensor_depth_truncation)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def step_cb(self, step):
        self._step = step

    def _build_scene_contraction(self):
        if not self.config.use_fg_aware_scene_contraction:
            self.scene_contraction = SceneContraction(
                order=float("inf") if self.config.scene_contraction_order == "inf" else None,
                scale_factor=self.config.scene_contraction_scale_factor,
            )
        else:
            alpha = self.config.fg_aware_scene_contraction_alpha
            CONSOLE.print(f"[bold yellow]Using ForegroundAwareSceneContraction({alpha=})")  # pyright: ignore
            self.scene_contraction = ForegroundAwareSceneContraction(
                order=float("inf") if self.config.scene_contraction_order == "inf" else None,
                alpha=alpha,
                aabb=self.scene_box.aabb,
                scale_factor=self.config.scene_contraction_scale_factor,
            )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        if self.config.background_model != "none":
            param_groups["field_background"] = list(self.field_background.parameters())
        else:
            param_groups["field_background"] = list(self.field_background)
        if self.config.bg_sampler_type == "proposal_network":
            param_groups["proposal_networks_bg"] = list(self.proposal_networks_bg.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.step_cb,
            )
        )

        if self.config.bg_sampler_type == "proposal_network" and self.config.use_proposal_weight_anneal_bg:
            # anneal the weights of the proposal network before doing PDF sampling
            N = 1000  # TODO：make configurable and tune
            proposal_weights_anneal_slope = 10.0

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, proposal_weights_anneal_slope)
                self.proposal_sampler_bg.set_anneal(anneal)

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
                    func=self.proposal_sampler_bg.step_cb,
                )
            )

        return callbacks

    def _build_collider(self) -> None:
        # Collider
        if self.scene_box.collider_type == "near_far":
            self.collider = NearFarCollider(near_plane=self.scene_box.near, far_plane=self.scene_box.far)
        elif self.scene_box.collider_type == "box":
            self.collider = AABBBoxCollider(self.scene_box, near_plane=self.scene_box.near)
        elif self.scene_box.collider_type == "box_near_far":
            self.collider = AABBBoxNearFarCollider(
                self.scene_box, near_plane=self.scene_box.near, far_plane=self.scene_box.far
            )
        elif self.scene_box.collider_type == "sphere":
            self.collider = SphereCollider(radius=1.0, near_plane=self.scene_box.near, far_plane=self.scene_box.far)
        else:
            raise NotImplementedError

        # command line near and far has highest priority
        if self.config.overwrite_near_far_plane:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

    def _build_bg_model_and_sampler(self) -> None:
        # background model
        if self.config.background_model == "grid":
            self.field_background = TCNNNerfactoField(
                self.scene_box.aabb,
                spatial_distortion=self.scene_contraction,
                num_images=self.num_train_data,
                use_appearance_embedding=self.config.bg_use_appearance_embedding,
                use_average_appearance_embedding=self.config.use_average_appearance_embedding,
                num_levels=self.config.bg_hash_grid_num_levels,
            )
        elif self.config.background_model == "mlp":
            position_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=9.0, include_input=True
            )
            direction_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
            )

            self.field_background = NeRFField(
                position_encoding=position_encoding,
                direction_encoding=direction_encoding,
                spatial_distortion=self.scene_contraction,
                spatial_normalization_region="full",
            )
        else:
            # dummy background model
            self.field_background = Parameter(torch.ones(1), requires_grad=False)

        # background sampler
        if self.config.bg_sampler_type == "lin_disp":
            self.sampler_bg = LinearDisparitySampler(num_samples=self.config.num_samples_outside)
        elif self.config.bg_sampler_type == "uniform_lin_disp":
            self.sampler_bg = UniformLinDispPiecewiseSampler(num_samples=self.config.num_samples_outside)
        elif self.config.bg_sampler_type == "proposal_network":
            assert isinstance(
                self.scene_contraction, ForegroundAwareSceneContraction
            )  # otherwise, setup a separate contraction for bg proposal nets
            self.density_fns_bg = []
            self.proposal_networks_bg = ModuleList()
            # assume using two proposal network with fixed hparams
            prop_net_args = [
                {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 64},
                {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256},
            ]
            proposal_warmup, proposal_update_every = 5000, 5
            num_proposal_samples_per_ray = self.config.num_bg_proposal_samples_per_ray
            for _args in prop_net_args:
                network = HashMLPDensityField(
                    self.scene_box.aabb,  # incorrect aabb as placeholder
                    spatial_distortion=self.scene_contraction,
                    spatial_normalization_region="full",
                    **_args,
                )
                self.proposal_networks_bg.append(network)
                self.density_fns_bg.append(network.density_fn)

            update_schedule = lambda step: np.clip(
                np.interp(step, [0, proposal_warmup], [0, proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )
            self.proposal_sampler_bg = ProposalNetworkSampler(
                num_nerf_samples_per_ray=self.config.num_samples_outside,
                num_proposal_samples_per_ray=num_proposal_samples_per_ray,
                num_proposal_network_iterations=2,
                use_uniform_sampler=False,  # use UniformLinDispPiecewiseSampler
                single_jitter=True,
                update_sched=update_schedule,
            )
        elif self.config.bg_sampler_type == "none":  # share sampler with fg
            self.sampler_bg = None
        else:
            raise ValueError(f"Unknown background sampler: {self.config.bg_sampler_type}")

    def _build_renderers(self) -> None:
        background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method=self.config.depth_renderer_mode)
        self.renderer_normal = SemanticRenderer()

    @abstractmethod
    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict:
        """_summary_
        ray points sampling and field inference of the foreground field.

        Args:
            ray_bundle (RayBundle): _description_
            return_samples (bool, optional): _description_. Defaults to False.
        """

    # @profiler.time_function
    def get_outputs(self, ray_bundle: RayBundle) -> Dict:
        outputs, samples_and_field_outputs = self.fg_fields_query_and_render(ray_bundle)
        outputs, samples_and_field_outputs_bg = self.bg_fields_query_and_render(
            ray_bundle, samples_and_field_outputs, outputs
        )
        self._update_outptus_for_vis(outputs, samples_and_field_outputs, samples_and_field_outputs_bg)

        # for auxilary field rendering (used by .get_output() methods of child classes)
        outputs.update(
            {
                "samples_and_field_outputs": samples_and_field_outputs,
                "samples_and_field_outputs_bg": samples_and_field_outputs_bg,
            }
        )

        return outputs

    def forward(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle.
        This outputs different things depending on the configuration of the model
        and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        outputs = self.get_outputs(ray_bundle)

        # remove unnecessary items to avoid memory leak
        for del_k in ["samples_and_field_outputs", "samples_and_field_outputs_bg"]:
            if del_k in outputs:
                del outputs[del_k]
        return outputs

    def fg_fields_query_and_render(self, ray_bundle: RayBundle) -> Dict[str, Any]:
        """query foreground fields and rendering"""
        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)

        if self.handle_no_intersect_rays:
            samples_and_field_outputs = self._handle_no_intersect_rays_fg(ray_bundle, samples_and_field_outputs)

        field_outputs = samples_and_field_outputs["field_outputs"]
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]
        bg_transmittance = samples_and_field_outputs["bg_transmittance"]

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        ray_dist = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        depth = ray_dist / ray_bundle.directions_norm

        accumulation = self.renderer_accumulation(weights=weights)

        if self.config.mask_depth_with_acc:
            depth[accumulation < 0.5] = 0.0  # for tsdf fusion to work properly

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "weights": weights,
            "directions_norm": ray_bundle.directions_norm,  # used to scale z_vals for free space and sdf loss
            "bg_transmittance": bg_transmittance,
        }

        sample_normals = field_outputs.get(FieldHeadNames.NORMAL, None)
        if sample_normals is not None:
            normal = self.renderer_normal(semantics=sample_normals, weights=weights)
            outputs.update({"normal": normal})

        self._update_outputs_for_loss(ray_bundle, outputs, samples_and_field_outputs)

        return outputs, samples_and_field_outputs

    def bg_fields_query_and_render(
        self, ray_bundle: RayBundle, samples_and_field_outputs_fg: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.config.background_model == "none" or not self.config.render_background:
            return outputs, {}

        ray_bundle = self._reset_near_far_for_background(ray_bundle)

        # bg field inference
        samples_and_field_outputs_bg = self.sample_and_forward_field_bg(
            ray_bundle, samples_and_field_outputs_fg, outputs
        )
        if self.handle_bottom_intersect_rays:  # TODO: ignore bg inference of rays intersecting the bottom plane
            self._handle_bottom_intersect_rays_bg(ray_bundle, samples_and_field_outputs_bg)
        weights_bg = samples_and_field_outputs_bg["weights"]
        ray_samples_bg = samples_and_field_outputs_bg["ray_samples"]

        # render background
        rgb_bg = self.renderer_rgb(rgb=samples_and_field_outputs_bg[FieldHeadNames.RGB], weights=weights_bg)

        depth_bg = self.renderer_depth(weights=weights_bg, ray_samples=ray_samples_bg)
        accumulation_bg = self.renderer_accumulation(weights=weights_bg)

        # merge background color to foregound color
        rgb = outputs["rgb"] + outputs["bg_transmittance"] * rgb_bg
        # TODO: update depth with background depth

        outputs.update(
            {
                "rgb": rgb,
                "bg_rgb": rgb_bg,
                "bg_accumulation": accumulation_bg,
                "bg_depth": depth_bg,
                "bg_weights": weights_bg,
            }
        )
        return outputs, samples_and_field_outputs_bg

    def sample_and_forward_field_bg(
        self, ray_bundle: RayBundle, samples_and_field_outputs_fg: Dict[str, Any], outputs: Dict[str, Any]
    ):
        if self.config.bg_sampler_type == "proposal_network":
            return self._sample_and_forward_field_bg_proposal(ray_bundle, samples_and_field_outputs_fg, outputs)

        ray_samples_bg = self.sampler_bg(ray_bundle)
        field_outputs_bg = self.field_background(ray_samples_bg)
        weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])

        samples_and_field_outputs_bg = {"ray_samples": ray_samples_bg, "weights": weights_bg, **field_outputs_bg}
        return samples_and_field_outputs_bg

    def _sample_and_forward_field_bg_proposal(
        self, ray_bundle: RayBundle, samples_and_field_outputs_fg: Dict[str, Any], outputs: Dict[str, Any]
    ):
        ray_samples_bg, weights_list_bg, ray_samples_list_bg = self.proposal_sampler_bg(
            ray_bundle, density_fns=self.density_fns_bg
        )
        field_outputs_bg = self.field_background(ray_samples_bg)
        weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])
        # NOTE: weights_bg doesn't take bg_transmittance into account (i.e., weights_bg *= bg_transmittance)
        # since the proposal_network_bg is unaware of the bg_transmittance
        weights_list_bg.append(weights_bg.clone())
        ray_samples_list_bg.append(ray_samples_bg)

        samples_and_field_outputs_bg = {
            "ray_samples": ray_samples_bg,
            "weights": weights_bg,
            "weights_list": weights_list_bg,
            "ray_samples_list": ray_samples_list_bg,
            **field_outputs_bg,
        }
        outputs.update(
            {
                "weights_list_bg": weights_list_bg,
                "ray_samples_list_bg": ray_samples_list_bg,
            }
        )
        return samples_and_field_outputs_bg

    def _update_outputs_for_loss(self, ray_bundle: RayBundle, outputs: Dict, samples_and_field_outputs: Dict):
        """update outputs for loss computation"""
        if self.training:  # compute gradients for eikonal loss
            field_outputs = samples_and_field_outputs["field_outputs"]
            if FieldHeadNames.GRADIENT in field_outputs:
                grad_points = field_outputs[FieldHeadNames.GRADIENT]
                outputs.update({"eik_grad": grad_points})

                # TODO volsdf use different point set for eikonal loss
                # grad_points = self.field.gradient(eik_points)
                # outputs.update({"eik_grad": grad_points})

            outputs.update({"curvature_losses": field_outputs.get("curvature_losses", None)})

            outputs.update(samples_and_field_outputs)

        if self.handle_no_intersect_rays:  # ignore loss computation of no-intersection rays
            no_intersect_mask = ray_bundle.metadata["no_intersect_mask"]  # [..., 1]
            outputs.update({"no_intersect_mask": no_intersect_mask})
        if self.handle_bottom_intersect_rays:  # ignore loss computation of rays intersecting the bottom plane
            outputs.update({"bottom_intersect_mask": ray_bundle.metadata["bottom_intersect_mask"]})

    def _update_outptus_for_vis(
        self, outputs: Dict, samples_and_field_outputs: Dict, samples_and_field_outputs_bg: Dict
    ) -> None:
        # proposal network depths
        self._compute_prop_net_depths(outputs, samples_and_field_outputs)
        self._compute_prop_net_depths(outputs, samples_and_field_outputs_bg, ouptut_key_prefix="prop_depth_bg")

        # accumulation -> mask
        outputs["mask"] = outputs["accumulation"] > 0.5

    def _compute_prop_net_depths(
        self, outputs: Dict, samples_and_field_outputs: Dict, ouptut_key_prefix: str = "prop_depth"
    ) -> None:
        """compute depths of proposal networks for visualization"""
        # TODO how can we move it to neus_facto without out of memory
        if "weights_list" not in samples_and_field_outputs:
            return

        weights_list = samples_and_field_outputs["weights_list"]
        ray_samples_list = samples_and_field_outputs["ray_samples_list"]
        for i in range(len(weights_list) - 1):
            outputs[f"{ouptut_key_prefix}_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i]
            )

    def _handle_no_intersect_rays_fg(self, ray_bundle: RayBundle, samples_and_field_outputs: Dict) -> Dict:
        """Handle rays not intersecting the foreground AABB.
        Assign 0 weights for all fg queries and 1 for bg_transmittance to these rays.

        Returns:
            samples_and_field_outputs: with zero weights for the rays not intersecting the foreground AABB.
        """
        weights = samples_and_field_outputs["weights"]  # [..., num_samples, 1]
        bg_transmittance = samples_and_field_outputs["bg_transmittance"]  # [..., 1]
        # ray_samples_list = samples_and_field_outputs["ray_samples_list"]
        num_samples = weights.shape[-2]
        no_intersect_mask = ray_bundle.metadata["no_intersect_mask"]  # [..., 1]

        bg_transmittance = torch.where(no_intersect_mask, torch.ones_like(bg_transmittance), bg_transmittance)
        no_intersect_mask = repeat(no_intersect_mask, "... m -> ... n m", n=num_samples)
        weights = torch.where(no_intersect_mask, torch.zeros_like(weights), weights)
        new_outputs = {"weights": weights, "bg_transmittance": bg_transmittance}

        if "weights_list" in samples_and_field_outputs:
            # TODO: delete this block since this might be meaningless, bacause rays without intersection are ignored
            # when computing the proposal loss
            weights_list = samples_and_field_outputs["weights_list"]
            weights_list.pop()
            weights_list.append(weights)
            new_outputs["weights_list"] = weights_list

        samples_and_field_outputs.update(new_outputs)
        return samples_and_field_outputs

    def _handle_bottom_intersect_rays_bg(self, ray_bundle: RayBundle, samples_and_field_outputs: Dict) -> Dict:
        """ignore ray samples in bg regions if the ray intersects the bottom plane."""
        weights = samples_and_field_outputs["weights"]
        num_samples = weights.shape[-2]
        bottom_intersect_mask = ray_bundle.metadata["bottom_intersect_mask"]
        bottom_intersect_mask = repeat(bottom_intersect_mask, "... m -> ... n m", n=num_samples)
        weights = torch.where(bottom_intersect_mask, torch.zeros_like(weights), weights)
        samples_and_field_outputs.update({"weights": weights})

    def _reset_near_far_for_background(self, ray_bundle: RayBundle) -> RayBundle:
        if self.handle_no_intersect_rays:
            # NOTE: self.config.far_plane_bg is not used, in favor of self.collider.far_plane
            _nears, _fars = ray_bundle.nears, ray_bundle.fars
            no_intersect_mask = ray_bundle.metadata["no_intersect_mask"]
            nears = torch.where(no_intersect_mask, _nears, _fars)
            fars = torch.where(no_intersect_mask, _fars, torch.full_like(_fars, self.collider.far_plane))
            ray_bundle.nears = nears
            ray_bundle.fars = fars
        else:
            ray_bundle.nears = ray_bundle.fars
            ray_bundle.fars = torch.ones_like(ray_bundle.fars) * self.config.far_plane_bg

        return ray_bundle

    def get_outputs_flexible(self, ray_bundle: RayBundle, additional_inputs: Dict[str, TensorType]) -> Dict:
        """run the model with additional inputs such as warping or rendering from unseen rays
        Args:
            ray_bundle: containing all the information needed to render that ray latents included
            additional_inputs: addtional inputs such as images, src_idx, src_cameras

        Returns:
            dict: information needed for compute gradients
        """
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        outputs = self.get_outputs(ray_bundle)

        ray_samples = outputs["ray_samples"]
        field_outputs = outputs["field_outputs"]

        if self.config.patch_warp_loss_mult > 0:
            # patch warping
            warped_patches, valid_mask = self.patch_warping(
                ray_samples,
                field_outputs[FieldHeadNames.SDF],
                field_outputs[FieldHeadNames.NORMAL],
                additional_inputs["src_cameras"],
                additional_inputs["src_imgs"],
                pix_indices=additional_inputs["uv"],
            )

            outputs.update({"patches": warped_patches, "patches_valid_mask": valid_mask})

        return outputs

    # @profiler.time_function
    def get_loss_dict(self, outputs, batch, metrics_dict=None, step=None) -> Dict:
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            # eikonal loss
            # TODO: additionally sample random points in the fg aabb and apply eikonal regularization
            # especially near to the border of the fg aabb.
            if "eik_grad" in outputs:
                grad_theta = outputs["eik_grad"]
                # handle out-of-box points
                if self.handle_no_intersect_rays:
                    no_intersect_mask = outputs["no_intersect_mask"].to(grad_theta.dtype)
                    n_intersect_pts = (no_intersect_mask.nelement() - no_intersect_mask.sum()) * np.prod(
                        grad_theta.shape[1:-1]
                    )
                    eikonal_loss = (
                        ((grad_theta.norm(2, dim=-1) - 1) ** 2) * (1 - no_intersect_mask)
                    ).sum() / n_intersect_pts
                    loss_dict["eikonal_loss"] = eikonal_loss * self.config.eikonal_loss_mult
                else:
                    loss_dict["eikonal_loss"] = (
                        (grad_theta.norm(2, dim=-1) - 1) ** 2
                    ).mean() * self.config.eikonal_loss_mult

            # foreground mask loss
            if "fg_mask" in batch and self.config.fg_mask_loss_mult > 0.0:
                fg_label = batch["fg_mask"].float().to(self.device)
                weights_sum = outputs["weights"].sum(dim=1).clip(1e-3, 1.0 - 1e-3)
                loss_dict["fg_mask_loss"] = (
                    F.binary_cross_entropy(weights_sum, fg_label) * self.config.fg_mask_loss_mult
                )

            # monocular normal loss
            if "normal" in batch and self.config.mono_normal_loss_mult > 0.0:
                if self._step > self.config.mono_normal_loss_warmup_steps:
                    normal_gt = batch["normal"].to(self.device)
                    normal_pred = outputs["normal"]
                    normal_mask = (outputs["accumulation"].detach().squeeze() > 0.95).float()
                    normal_loss_scale = map_range_val(self._step, 
                                                    self.config.mono_normal_loss_warmup_steps, 
                                                    self.config.mono_normal_loss_steps, 
                                                    1, 
                                                    self.config.normal_loss_final_scale, 
                                                    type='log', factor=1.)

                    normal_loss = monosdf_normal_loss(normal_pred * normal_mask[...,None], normal_gt * normal_mask[...,None])       
                    loss_dict["normal_loss"] = normal_loss * self.config.mono_normal_loss_mult * normal_loss_scale

            # monocular depth loss
            if "depth" in batch and self.config.mono_depth_loss_mult > 0.0:
                # TODO check it's true that's we sample from only a single image
                # TODO only supervised pixel that hit the surface and remove hard-coded scaling for depth
                depth_gt = batch["depth"].to(self.device)[..., None]
                depth_pred = outputs["depth"]

                mask = torch.ones_like(depth_gt).reshape(1, 32, -1).bool()
                loss_dict["depth_loss"] = (
                    self.depth_loss(depth_pred.reshape(1, 32, -1), (depth_gt * 50 + 0.5).reshape(1, 32, -1), mask)
                    * self.config.mono_depth_loss_mult
                )

            # sensor depth loss
            if "sensor_depth" in batch and (
                self.config.sensor_depth_l1_loss_mult > 0.0
                or self.config.sensor_depth_freespace_loss_mult > 0.0
                or self.config.sensor_depth_sdf_loss_mult > 0.0
            ):
                l1_loss, free_space_loss, sdf_loss = self.sensor_depth_loss(batch, outputs)

                loss_dict["sensor_l1_loss"] = l1_loss * self.config.sensor_depth_l1_loss_mult
                loss_dict["sensor_freespace_loss"] = free_space_loss * self.config.sensor_depth_freespace_loss_mult
                loss_dict["sensor_sdf_loss"] = sdf_loss * self.config.sensor_depth_sdf_loss_mult

            # multi-view photoconsistency loss as Geo-NeuS
            if "patches" in outputs and self.config.patch_warp_loss_mult > 0.0:
                patches = outputs["patches"]
                patches_valid_mask = outputs["patches_valid_mask"]

                loss_dict["patch_loss"] = (
                    self.patch_loss(patches, patches_valid_mask) * self.config.patch_warp_loss_mult
                )

            # sparse points sdf loss
            if "sparse_sfm_points" in batch and self.config.sparse_points_sdf_loss_mult > 0.0:
                sparse_sfm_points = batch["sparse_sfm_points"].to(self.device)
                sparse_sfm_points_sdf = self.field.forward_geonetwork(sparse_sfm_points)[:, 0].contiguous()
                loss_dict["sparse_sfm_points_sdf_loss"] = (
                    torch.mean(torch.abs(sparse_sfm_points_sdf)) * self.config.sparse_points_sdf_loss_mult
                )

            # total variational loss for multi-resolution periodic feature volume
            if self.config.periodic_tvl_mult > 0.0:
                assert self.field.config.encoding_type == "periodic"
                loss_dict["tvl_loss"] = self.field.encoding.get_total_variation_loss() * self.config.periodic_tvl_mult

            self._compute_curvature_loss(outputs, loss_dict)
            self._compute_interlevel_loss_bg(outputs, loss_dict)
            self._compute_ptcd_reg_fg(outputs, loss_dict)
            self._compute_ptcd_reg_plane(outputs, loss_dict)

        return loss_dict

    # @profiler.time_function
    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        images_dict = {}
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        if "depth" in batch:
            depth_gt = batch["depth"].to(self.device)
            depth_pred = outputs["depth"]

            # align to predicted depth and normalize
            scale, shift = compute_scale_and_shift(
                depth_pred[None, ..., 0], depth_gt[None, ...], depth_gt[None, ...] > 0.0
            )
            depth_pred = depth_pred * scale + shift

            combined_depth = torch.cat([depth_gt[..., None], depth_pred], dim=1)
            combined_depth = colormaps.apply_depth_colormap(combined_depth)
        else:
            # TODO: set depth[accumulation < 0.1] = depth[accumulation >= 0.1].min()
            depth = colormaps.apply_depth_colormap(
                outputs["depth"],
                accumulation=outputs["accumulation"],
            )
            combined_depth = torch.cat([depth], dim=1)
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        if "normal" in outputs:
            normal = outputs["normal"]
            # don't need to normalize here
            # normal = torch.nn.functional.normalize(normal, p=2, dim=-1)
            normal = (normal + 1.0) / 2.0
            if "normal" in batch:
                normal_gt = (batch["normal"].to(self.device) + 1.0) / 2.0
                combined_normal = torch.cat([normal_gt, normal], dim=1)
            else:
                combined_normal = torch.cat([normal], dim=1)
            images_dict.update({"normal": combined_normal})

        if "sensor_depth" in batch:
            sensor_depth = batch["sensor_depth"]
            depth_pred = outputs["depth"]

            combined_sensor_depth = torch.cat([sensor_depth[..., None], depth_pred], dim=1)
            combined_sensor_depth = colormaps.apply_depth_colormap(combined_sensor_depth)
            images_dict["sensor_depth"] = combined_sensor_depth

        if self.handle_no_intersect_rays:
            self._get_no_intersection_ray_mask(outputs, images_dict)

        if self.config.bg_sampler_type == "proposal_network":
            for i in range(self.config.num_proposal_iterations):
                key = f"prop_depth_bg_{i}"
                # TODO: set depth[accumulation < 0.1] = depth[accumulation >= 0.1].min()
                prop_depth_i = colormaps.apply_depth_colormap(outputs[key])
                images_dict[key] = prop_depth_i

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        return metrics_dict, images_dict

    def _get_no_intersection_ray_mask(self, outputs: Dict[str, torch.Tensor], images_dict: Dict[str, torch.Tensor]):
        """visualizing ray-box intersection results"""
        no_intersect_mask = outputs["no_intersect_mask"].to(torch.float32)  # (h, w, 1)
        images_dict["no_intersect_mask"] = no_intersect_mask

    def _compute_curvature_loss(self, outputs: Dict[str, torch.Tensor], loss_dict: Dict[str, torch.Tensor]):
        """compute curvature loss"""
        if (
            self.config.curvature_loss_mult == 0.0
            or self.config.curvature_loss_n_iters is not None
            and self._step > self.config.curvature_loss_n_iters
        ):
            return

        curvature_losses = outputs["curvature_losses"]  # (n_rays, n_pts, 1)
        if self.handle_no_intersect_rays:
            no_intersect_mask = outputs["no_intersect_mask"].to(curvature_losses.dtype)
            n_intersect_pts = (no_intersect_mask.nelement() - no_intersect_mask.sum()) * np.prod(
                curvature_losses.shape[1:-1]
            )
            curvature_loss = (curvature_losses * (1 - no_intersect_mask[:, None])).sum() / n_intersect_pts
        else:
            curvature_loss = torch.mean(curvature_losses)
        loss_dict["curvature_loss"] = curvature_loss * self.config.curvature_loss_mult
        return

    def _compute_interlevel_loss_bg(self, outputs: Dict[str, torch.Tensor], loss_dict: Dict[str, torch.Tensor]):
        """compute proposal network loss for bg sampler"""
        if not self.training or self.config.bg_sampler_type != "proposal_network":
            return

        bg_interlevel_loss_mult = 1.0
        weights_list, ray_samples_list = self._get_interlevel_loss_inputs_bg(outputs)
        interlevel_loss_val = interlevel_loss(weights_list, ray_samples_list)
        if torch.isnan(interlevel_loss_val):
            __import__("ipdb").set_trace()
        loss_dict["bg_interlevel_loss"] = bg_interlevel_loss_mult * interlevel_loss_val

    def _get_interlevel_loss_inputs_bg(self, outputs: Dict[str, torch.Tensor]):
        weights_list = outputs["weights_list_bg"]  # (n_rays, n_samples, 1)
        ray_samples_list = outputs["ray_samples_list_bg"]  # (n_rays, n_samples)

        if not self.handle_bottom_intersect_rays:
            return weights_list, ray_samples_list

        # ignore supervision of rays intersecting the bottom plane of the fg aabb
        # we can optionally supervise the underneath region with zero weights
        bottom_intersect_mask = outputs["bottom_intersect_mask"][..., 0]  # (n_rays, )
        weights_list = [weights[~bottom_intersect_mask] for weights in weights_list]
        ray_samples_list = [ray_samples[~bottom_intersect_mask] for ray_samples in ray_samples_list]

        return weights_list, ray_samples_list

    def _build_anneal_fn(self, anneal_fn_name, max_step):
        if max_step is None:
            ptcd_reg_disabled = self.config.ptcd_reg_plane_mult == 0.0 and self.config.ptcd_reg_fg_mult == 0.0
            assert ptcd_reg_disabled or anneal_fn_name == "constant"
            return lambda x: 1.0

        if anneal_fn_name == "cosine":
            anneal_fn = scheduler.cosine_annealing
        elif anneal_fn_name == "exponential":
            anneal_fn = scheduler.exp_annealing
        elif anneal_fn_name == "constant":
            anneal_fn = scheduler.constant
        else:
            raise ValueError(f"Unknown anneal fn: {anneal_fn_name}")
        return partial(anneal_fn, max_step=max_step, min_val=0.0, max_val=1.0)

    def _compute_ptcd_reg_fg(self, outputs: Dict[str, torch.Tensor], loss_dict: Dict[str, torch.Tensor]):
        """compute foreground point cloud regularization loss
        use point-wise threshold by default.
        """
        if self.config.ptcd_reg_fg_mult == 0.0 or self._step > self.config.ptcd_reg_n_iters:
            return

        anneal_weight = self.ptcd_reg_anneal_fn(self._step)
        writer.put_scalar(name="ptcd_reg_fg_anneal_weight", scalar=anneal_weight, step=self._step)
        fg_pts, fg_sdf_thrs = self.ptcd_data.sample_fg_points(self.config.ptcd_reg_fg_n_pts_per_iter, replace=True)
        fg_pts_sdf = self.field.forward_geonetwork(fg_pts)[:, 0].contiguous()
        # only support point-wise threshold for now
        fg_sdf_loss = -torch.min(fg_sdf_thrs - torch.abs(fg_pts_sdf), torch.zeros_like(fg_pts[:, 0])).mean()
        loss_dict["ptcd_reg_fg"] = fg_sdf_loss * self.config.ptcd_reg_fg_mult * anneal_weight

    def _compute_ptcd_reg_plane(self, outputs: Dict[str, torch.Tensor], loss_dict: Dict[str, torch.Tensor]):
        """compute plane point cloud regularization loss"""
        if (
            self.config.ptcd_reg_plane_mult == 0.0
            or self._step > self.config.ptcd_reg_n_iters
            or len(self.ptcd_data.plane_pts) == 0  # some sequence might not have plane pts
        ):
            return

        anneal_weight = self.ptcd_reg_anneal_fn(self._step)
        writer.put_scalar(name="ptcd_reg_plane_anneal_weight", scalar=anneal_weight, step=self._step)
        # sample plane pts (optionally sample half by random sampling from parametric plane)
        plane_pts, udf_plane_to_fg, _ = self.ptcd_data.sample_plane_points(
            self.config.ptcd_reg_plane_n_pts_per_iter, replace=True
        )
        plane_pts_sdf = self.field.forward_geonetwork(plane_pts)[:, 0].contiguous()
        plane_sdf_loss = torch.max(udf_plane_to_fg - torch.abs(plane_pts_sdf), torch.zeros_like(udf_plane_to_fg)).mean()
        loss_dict["ptcd_reg_plane"] = plane_sdf_loss * self.config.ptcd_reg_plane_mult * anneal_weight

    def _compute_mask_beta_prior(self, outputs: Dict[str, torch.Tensor], loss_dict: Dict[str, torch.Tensor]):
        """compute mask beta prior regularization"""
        raise NotImplementedError
