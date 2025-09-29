"""
NeuSFacto w/ explicit point cloud regularization for scene decomposition.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import numpy as np
import torch
from rich.console import Console
from torch.nn import Parameter
from torch.nn import functional as F
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.fields.sdf_field import SDFField, SDFFieldConfig
from nerfstudio.model_components.ray_samplers import NeuSSampler, UniformSampler
from nerfstudio.models.neus_facto import NeuSFactoModel, NeuSFactoModelConfig
from nerfstudio.utils import colormaps, writer

CONSOLE = Console(width=120)


@dataclass
class NeuSFactoRegModelConfig(NeuSFactoModelConfig):
    """NeusFactoReg Model Config"""

    _target: Type = field(default_factory=lambda: NeuSFactoRegModel)

    plane_field_type: Literal["nerf", "sdf"] = "nerf"
    plane_sdf_field: SDFFieldConfig = SDFFieldConfig(
        spatial_normalization_region="aabb",
        use_normalized_raw_coords=False,  # False might cause bad geo-init, all initial sdf are large and leads to low weights
        use_position_encoding=False,
        use_grid_feature=True,
        hash_grid_num_levels=14,  # 5
        hash_grid_log2_hashmap_size=17,
        hash_grid_max_res=2048,
        num_layers=1,
        num_layers_color=2,
        hidden_dim=64,  # 32
        geo_feat_dim=15,
        direction_encoding_type="sh",
        hidden_dim_color=64,  # 32
        color_network_include_sdf=False,
        bias=0.5,  # 0.5 -> 0.1  TODO: initialize sdf_field as a plane
        beta_init=0.3,
        use_appearance_embedding=False,
        inside_outside=False,
        weight_norm=False,
    )
    plane_height_ratio: float = 0.1
    """ratio of the scene box height to use as the height of plane aabb"""
    num_samples_plane: int = 16
    plane_hierarchical_sampling: bool = False
    """number of samples to use for plane sdf field"""
    ptcd_reg_plane_field_mult: float = 0.0
    ptcd_reg_plane_field_n_pts_per_iter: int = 5000


class NeuSFactoRegModel(NeuSFactoModel):
    """NeusFacto w/ explicit point cloud regularization and separate modeling of the ground plane for scene decomposition."""

    config: NeuSFactoRegModelConfig

    def populate_modules(self):
        super().populate_modules()

        # build aabb for plane region
        scene_box_height = (self.scene_box.aabb[1, 1] - self.scene_box.aabb[0, 1]).abs()
        self.plane_box_height = scene_box_height * self.config.plane_height_ratio
        plane_aabb = self.scene_box.aabb.clone()
        plane_aabb[0, 1] = plane_aabb[1, 1] - self.plane_box_height  # assume +y points downward
        # TODO: use the exact plane center and smaller plane bbox extent

        # build plane field
        if self.config.plane_field_type == "sdf":
            self.field_plane = self.config.plane_sdf_field.setup(
                aabb=plane_aabb,
                spatial_distortion=self.scene_contraction,
                num_images=self.num_train_data,
                use_average_appearance_embedding=self.config.use_average_appearance_embedding,
                compute_curvature_loss=self.config.curvature_loss_mult > 0,
            )
        elif self.config.plane_field_type == "nerf":
            self.field_plane = TCNNNerfactoField(
                plane_aabb,
                spatial_distortion=self.scene_contraction,
                num_images=self.num_train_data,
                use_appearance_embedding=self.config.bg_use_appearance_embedding,
                use_average_appearance_embedding=self.config.use_average_appearance_embedding,
                num_levels=16,
                max_res=256,
                log2_hashmap_size=17,
                hidden_dim=32,
                num_layers_color=3,
                hidden_dim_color=32,
            )
        else:
            raise ValueError(f"Unknown plane field type: {self.config.plane_field_type}")

        # build plane ray sampler
        if not self.config.plane_hierarchical_sampling:
            self.sampler_plane = UniformSampler(num_samples=self.config.num_samples_plane)
        else:
            assert self.config.plane_field_type == "sdf"
            self.sampler_plane = NeuSSampler(
                num_samples=self.config.num_samples_plane * 2,
                num_samples_importance=self.config.num_samples_plane,
                num_upsample_steps=4,
            )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["field_plane"] = list(self.field_plane.parameters())
        return param_groups

    def sample_and_forward_field(self, ray_bundle: RayBundle):
        samples_and_field_outputs = super().sample_and_forward_field(ray_bundle)

        # sample additional points on top of the bottom plane
        mask = ray_bundle.metadata["bottom_intersect_mask"][..., 0]  # (n,)
        if not mask.any():
            return samples_and_field_outputs

        ray_bundle_plane = deepcopy(ray_bundle)[mask]
        ray_d, far, near = ray_bundle_plane.directions, ray_bundle_plane.fars, ray_bundle_plane.nears  # (n, *)
        y_dir = ray_d.new_tensor([[0.0, 1.0, 0.0]])
        _cos = (ray_d * y_dir).sum(-1) / ray_d.norm(dim=-1)
        ray_dist_offset = self.plane_box_height / _cos[..., None]  # (n, 1)
        new_near = far - ray_dist_offset
        ray_bundle_plane.nears = torch.max(new_near, near)

        plane_field_outputs = {}
        if self.config.plane_hierarchical_sampling:
            ray_samples_plane = self.sampler_plane(ray_bundle_plane, self.field_plane.get_sdf)
        else:
            ray_samples_plane = self.sampler_plane(ray_bundle_plane)

        if self.config.plane_field_type == "sdf":
            field_outputs = self.field_plane(ray_samples_plane, return_alphas=True)
            weights, transmittance = ray_samples_plane.get_weights_and_transmittance_from_alphas(
                field_outputs[FieldHeadNames.ALPHA]
            )  # (n_bottom, n_samples_plane, 1)
            plane_field_outputs.update(
                {
                    "plane_normals": field_outputs[FieldHeadNames.NORMAL],
                    "plane_eik_grad": field_outputs[FieldHeadNames.GRADIENT],
                }
            )
        else:  # nerf
            field_outputs = self.field_plane(ray_samples_plane)
            densities = field_outputs[FieldHeadNames.DENSITY]
            weights, transmittance = ray_samples_plane.get_weights_and_transmittance(densities)

        samples_and_field_outputs.update(
            {
                **plane_field_outputs,
                "plane_ray_samples": ray_samples_plane,
                "plane_weights": weights,
                "plane_transmittance": transmittance[:, -1, :],
                "plane_rgbs": field_outputs[FieldHeadNames.RGB],
            }
        )

        return samples_and_field_outputs

    def fg_fields_query_and_render(self, ray_bundle: RayBundle) -> Dict[str, Any]:
        outputs, samples_and_field_outputs = super().fg_fields_query_and_render(ray_bundle)
        if not self.config.render_background:
            return outputs, samples_and_field_outputs

        # render plane field
        mask = ray_bundle.metadata["bottom_intersect_mask"][..., 0]  # (n,)
        normal = torch.zeros_like(outputs["normal"])  # vis
        depth = torch.zeros_like(outputs["depth"])  # vis
        acc = torch.zeros_like(outputs["accumulation"])
        plane_rgb = torch.ones_like(outputs["rgb"])
        if not mask.any():
            outputs.update(
                {"plane_normal": normal, "plane_depth": depth, "plane_accumulation": acc, "plane_rgb": plane_rgb}
            )
            return outputs, samples_and_field_outputs

        # rendering the plane field
        plane_weights, plane_trans, plane_rgbs = map(
            lambda x: samples_and_field_outputs.get(x), ["plane_weights", "plane_transmittance", "plane_rgbs"]
        )
        _plane_rgb = self.renderer_rgb(rgb=plane_rgbs, weights=plane_weights)

        # composite with the fg rendering
        fg_rgb, fg_trans = outputs["rgb"][mask], outputs["bg_transmittance"][mask]
        _fg_rgb = fg_rgb + fg_trans * _plane_rgb
        _bg_trans = fg_trans * plane_trans
        bg_trans = torch.ones_like(outputs["bg_transmittance"])
        fg_rgb = torch.zeros_like(outputs["rgb"])
        bg_trans[mask] = _bg_trans
        fg_rgb[mask] = _fg_rgb

        # plane normals (& depths) rendering
        if self.config.plane_field_type == "sdf":
            plane_normals = samples_and_field_outputs["plane_normals"]
            _normal = self.renderer_normal(semantics=plane_normals, weights=plane_weights)
            normal[mask] = _normal

        plane_ray_samples = samples_and_field_outputs["plane_ray_samples"]
        _depth = self.renderer_depth(weights=plane_weights, ray_samples=plane_ray_samples)
        _accumulation = self.renderer_accumulation(weights=plane_weights)
        depth[mask], acc[mask], plane_rgb[mask] = _depth, _accumulation, _plane_rgb

        outputs.update(
            {"plane_normal": normal, "plane_depth": depth, "plane_accumulation": acc, "plane_rgb": plane_rgb}
        )

        # update
        if self.training and self.config.plane_field_type == "sdf":
            _plane_eik_grad = samples_and_field_outputs["plane_eik_grad"]
            plane_eik_grad = _plane_eik_grad.new_zeros((normal.shape[0], _plane_eik_grad.shape[-2], 3))
            plane_eik_grad[mask] = _plane_eik_grad
            outputs["plane_eik_grad"] = plane_eik_grad
            outputs["plane_mask"] = mask  # for loss computation
        outputs["bg_transmittance"] = torch.where(mask[..., None], bg_trans, outputs["bg_transmittance"])
        outputs["rgb"] = torch.where(mask[..., None].repeat(1, 3), fg_rgb, outputs["rgb"])
        return outputs, samples_and_field_outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None, step=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # eikonal loss for plane sdf field
        if "plane_eik_grad" in outputs:
            grad_theta = outputs["plane_eik_grad"]
            plane_mask = outputs["plane_mask"][..., None].to(grad_theta.dtype)
            n_pts = plane_mask.sum() * np.prod(grad_theta.shape[1:-1])
            eikonal_loss = (((grad_theta.norm(2, dim=-1) - 1) ** 2) * plane_mask).sum() / n_pts
            loss_dict["plane_eikonal_loss"] = (
                eikonal_loss * self.config.eikonal_loss_mult
            )  # donnot use a separate weight for plane eikonal loss

        self._compute_ptcd_reg_plane_field(outputs, loss_dict)

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        # plane normal
        if "plane_normal" in outputs:
            images_dict.update({"plane_normal": (outputs["plane_normal"] + 1.0) / 2.0})
        if "plane_depth" in outputs:
            depth = colormaps.apply_depth_colormap(
                outputs["plane_depth"],
                accumulation=outputs["plane_accumulation"],
            )
            acc = colormaps.apply_colormap(outputs["plane_accumulation"])
            images_dict.update({"plane_depth": depth, "plane_accumulation": acc})
        return metrics_dict, images_dict

    def _compute_ptcd_reg_plane_field(self, outputs: Dict[str, torch.Tensor], loss_dict: Dict[str, torch.Tensor]):
        """compute point cloud regularization loss on the plane sdf field"""
        if self.config.ptcd_reg_plane_field_mult == 0.0 or self._step > self.config.ptcd_reg_n_iters:
            return

        anneal_weight = self.ptcd_reg_anneal_fn(self._step)
        writer.put_scalar(name="ptcd_reg_plane_field_anneal_weight", scalar=anneal_weight, step=self._step)
        plane_pts, _, plane_sdf_thrs = self.ptcd_data.sample_plane_points(
            self.config.ptcd_reg_plane_field_n_pts_per_iter, replace=True
        )
        plane_pts_sdf = self.field_plane.forward_geonetwork(plane_pts)[:, 0].contiguous()
        # only support point-wise threshold for now
        plane_sdf_loss = -torch.min(plane_sdf_thrs - torch.abs(plane_pts_sdf), torch.zeros_like(plane_pts[:, 0])).mean()
        loss_dict["ptcd_reg_plane_field"] = plane_sdf_loss * self.config.ptcd_reg_plane_field_mult * anneal_weight

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training and isinstance(self.field_plane, SDFField):
            # training statics
            metrics_dict["s_val_plane"] = self.field_plane.deviation_network.get_variance().item()
            metrics_dict["inv_s_plane"] = 1.0 / self.field_plane.deviation_network.get_variance().item()

        return metrics_dict
