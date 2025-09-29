"""
NeuSFacto + DistilledFeatureField
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Type, Tuple, Dict, Optional
from torchtyping import TensorType
from collections import defaultdict
from tqdm import tqdm

import torch
from rich.console import Console
from torch.nn import Parameter, MSELoss
from torch.nn import functional as F
from sklearn.decomposition import PCA
from einops import rearrange, repeat

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.feature_field import FeatureFieldConfig, FeatureSegFieldConfig
from nerfstudio.model_components.renderers import FeatureRenderer, SemanticRenderer
from nerfstudio.model_components.ray_samplers import RayTracingSampler
from nerfstudio.models.neus_facto import NeuSFactoModel, NeuSFactoModelConfig
from nerfstudio.model_components.ray_samplers import Sampler, SpacedSampler, DummySampler
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.utils.pointclouds import BasicPointClouds
from nerfstudio.utils.colormaps import apply_colormap, apply_depth_colormap
from nerfstudio.utils.mask_utils import overlay_mask_on_image

CONSOLE = Console(width=120)


@dataclass
class NeuSFactoDFFModelConfig(NeuSFactoModelConfig):
    """NeusFactoDFF Model Config"""

    _target: Type = field(default_factory=lambda: NeuSFactoDFFModel)

    feature_field: FeatureFieldConfig = FeatureFieldConfig()
    feature_field_scene_contraction_scale_factor: float = 1.0
    """use a separate scene contraction for feature field, since the feature field use a simple field
    to represent the whole scene, including both fg and bg.
    """
    feature_loss_mult: float = 0.1  # NOTE: this multiplier is meaningless if all other parts are frozen
    
    render_fg_seg: bool = True
    """rendering foreground segmentation masks during evaluation"""
    feature_seg_field: FeatureSegFieldConfig = FeatureSegFieldConfig()
    """segmentation field config"""
    fg_seg_binary_thr: float = 0.5
    """binarization threshold for 2d segmentation masks"""
    fg_3d_seg_binary_thr: float = 0.5
    """binarization threshold for 3d segmentation masks"""
    sdf_rectification_value: float = 10.0
    """rectify absolute SDF values of positions where the segmentation field equals 0 to this value"""
    
    use_surface_rendering: bool = False
    """learn the feature field with surface rendering"""
    # TODO: RayTracingSampler configs
    ignore_false_positive_sphere_tracing_with_accumulation: bool = True
    """sphere tracing might produce some incorrect intersections near to the border of
    fg bboxes because sdf here are invalid, optionally filter out theses incorrect intersections
    whose accumulation is less than 0.5
    """
    complement_false_negative_sphere_tracing_with_volrend: bool = True
    """sphere tracing might fail to find some intersections, use depth of volume rendering
    as pseudo intersections.
    """
    

class NeuSFactoDFFModel(NeuSFactoModel):
    """NeuS facto model

    Args:
        config: NeuSFactoDFFModel configuration to instantiate model
    """

    config: NeuSFactoDFFModelConfig

    def populate_modules(self):
        """Set feature fields for foreground and background"""
        super().populate_modules()
        self.ptcd_data: BasicPointClouds = self.kwargs["metadata"]["ptcd_data"]
        self.ptcd_data.build_nn_search_index(self.config.feature_seg_field.separate_bg_plane)
        
        self.feature_field_scene_contraction = SceneContraction(
            order=float("inf"),
            scale_factor=self.config.feature_field_scene_contraction_scale_factor
        )
        self.feature_field = self.config.feature_field.setup(
            spatial_distortion=self.feature_field_scene_contraction
        )
        self.feature_seg_field = self.config.feature_seg_field.setup(ptcd_data=self.ptcd_data,
                                                                     aabb=self.scene_box.aabb)
        self.renderer_depth_median = DepthRenderer("median")
        self.renderer_feature = FeatureRenderer()
        self.renderer_fg_seg = SemanticRenderer()
        self.feature_loss = MSELoss()
        
        if self.config.use_surface_rendering:
            self.ray_tracing_sampler = RayTracingSampler(aabb=self.scene_box.aabb)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        for param_group in param_groups.values():
            for param in param_group:
                param.requires_grad_(False)
        CONSOLE.print("All parameters except for feature field have been frozen.")

        param_groups["feature_field"] = list(self.feature_field.parameters())

        return param_groups

    def sample_and_forward_field(self, ray_bundle: RayBundle):
        # if not (self.training and self.config.use_surface_rendering):
        #     samples_and_field_outputs = super().sample_and_forward_field(ray_bundle)
        #     ray_samples = samples_and_field_outputs["ray_samples"]
        # else:
        #     # TODO: not working yet (need to udpate SurfaceModel to ignore fg rendering)
        #     # during training, we can avoid the ``super().sample_and_forward_field(ray_bundle)`` call
        #     # to speed up inference, since we only need to render the feature field
        #     samples_and_field_outputs = {}
        #     ray_samples = None
        
        samples_and_field_outputs = super().sample_and_forward_field(ray_bundle)
        volrend_ray_samples = samples_and_field_outputs["ray_samples"]
        volrend_weights = samples_and_field_outputs["weights"]
            
        if self.config.use_surface_rendering:
            samples_and_field_outputs_rt = self._sample_and_forward_field_ray_tracing(
                ray_bundle, volrend_ray_samples, volrend_weights
            )
            samples_and_field_outputs.update(samples_and_field_outputs_rt)  # "{xxx}_st"
            ray_samples = samples_and_field_outputs_rt["ray_samples_rt"]
        else:
            ray_samples = volrend_ray_samples
        
        feature_field_outputs = self.feature_field(ray_samples)
        samples_and_field_outputs.update(feature_field_outputs)
        
        if not self.training and self.config.render_fg_seg:
            feature_field_seg_outptus = self.feature_seg_field(
                ray_samples, feature_field_outputs[FieldHeadNames.FEATURE]
            )
            samples_and_field_outputs.update(feature_field_seg_outptus)

        return samples_and_field_outputs

    def _sample_and_forward_field_ray_tracing(
        self,
        ray_bundle: RayBundle,
        volrend_ray_samples: RaySamples,
        volrend_weights: TensorType["bs": ..., 1],
    ):
        sdf_fn = lambda x: self.field.forward_geonetwork(x)[..., [0]]  # (..., 3) -> (..., 1)
        ray_samples, non_convergent_masks = self.ray_tracing_sampler(ray_bundle, sdf_fn)  # (..., 1, 3), (..., 1, 1)
        non_convergent_masks_raw = non_convergent_masks.clone()

        if (
            self.config.ignore_false_positive_sphere_tracing_with_accumulation or
            self.config.complement_false_negative_sphere_tracing_with_volrend
        ):
            if ray_bundle.metadata.get("no_intersect_mask", None) is not None:
                no_intersect_mask = ray_bundle.metadata["no_intersect_mask"]
                no_intersect_mask = repeat(no_intersect_mask, "... m -> ... n m", n=volrend_weights.shape[-2])
                volrend_weights = torch.where(no_intersect_mask, torch.zeros_like(volrend_weights), volrend_weights)
            volrend_acc = self.renderer_accumulation(weights=volrend_weights)
            acc_bg_mask = volrend_acc < 0.5
        if self.config.ignore_false_positive_sphere_tracing_with_accumulation:
            non_convergent_masks |= acc_bg_mask[..., None]
        
        if self.config.complement_false_negative_sphere_tracing_with_volrend:
            volrend_ray_dist = self.renderer_depth_median(
                weights=volrend_weights, ray_samples=volrend_ray_samples
            )[..., None]
            false_negative_mask = non_convergent_masks & ~acc_bg_mask[..., None]
            non_convergent_masks[false_negative_mask] = False
            ray_samples.frustums.starts[false_negative_mask] = volrend_ray_dist[false_negative_mask]
            ray_samples.frustums.ends[false_negative_mask] = volrend_ray_dist[false_negative_mask]
            
        # assign weight = 1.0 for the single ray sample (intersection pt)
        # equivalent w/ self._handle_no_intersect_rays()
        weights = torch.where(non_convergent_masks,
                              torch.zeros_like(ray_samples.frustums.starts),
                              torch.ones_like(ray_samples.frustums.starts))  # (..., 1, 1)
        bg_transmittance = torch.where(non_convergent_masks[..., 0, [0]],
                                       torch.ones_like(weights[..., 0, [0]]),
                                       torch.zeros_like(weights[..., 0, [0]]))  # (..., 1)
        
        # ray tracing depth
        depth = ray_samples.frustums.starts[..., 0, :] / ray_bundle.directions_norm  # starts = ends
        
        samples_and_field_outputs = {
            "non_convergent_mask_rt": non_convergent_masks,
            "non_convergent_mask_rt_raw": non_convergent_masks_raw,  # w/o postprocessing
            "ray_samples_rt": ray_samples,
            "weights_rt": weights,
            "bg_transmittance_rt": bg_transmittance,
            "depth_rt": depth
        }
        return samples_and_field_outputs
    
    def get_outputs(self, ray_bundle: RayBundle) -> Dict:
        outputs = super().get_outputs(ray_bundle)
        
        samples_and_field_outputs = outputs["samples_and_field_outputs"]
        _weights_key = "weights" if not self.config.use_surface_rendering else "weights_rt"
        _trans_key = "bg_transmittance" if not self.config.use_surface_rendering else "bg_transmittance_rt"
        weights = samples_and_field_outputs[_weights_key]
        bg_transmittance = samples_and_field_outputs[_trans_key]

        # fg rendering (field query is done in sample_and_forward_field)
        feature = self.renderer_feature(
            features=samples_and_field_outputs[FieldHeadNames.FEATURE], weights=weights
        )

        if not self.training and self.config.render_fg_seg:
            with torch.no_grad():
                fg_seg_prob = self.renderer_fg_seg(
                    samples_and_field_outputs[FieldHeadNames.FG_SEG].detach(), weights=weights.detach()
                )
            outputs["fg_seg_prob"] = fg_seg_prob

        # bg field qeury and rendering
        if self.config.background_model != "none":
            samples_and_field_outputs_bg = outputs["samples_and_field_outputs_bg"]
            ray_samples_bg = samples_and_field_outputs_bg["ray_samples"]
            weights_bg = samples_and_field_outputs_bg["weights"]
            feature_field_outputs_bg = self.feature_field(ray_samples_bg)
            feature_bg = self.renderer_feature(
                features=feature_field_outputs_bg[FieldHeadNames.FEATURE], weights=weights_bg
            )
            # composition
            feature = feature + bg_transmittance * feature_bg

        outputs["feature"] = feature
        if self.config.use_surface_rendering:
            outputs["non_convergent_mask_rt"] = samples_and_field_outputs["non_convergent_mask_rt"]
            outputs["non_convergent_mask_rt_raw"] = samples_and_field_outputs["non_convergent_mask_rt_raw"]
            outputs["depth_rt"] = samples_and_field_outputs["depth_rt"]
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training and self.config.feature_loss_mult > 0.:
            loss_dict["feature_loss"] = (
                self.feature_loss(batch["feature"], outputs["feature"]) *
                self.config.feature_loss_mult
            )

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        self._get_pca_feature_rendering(outputs, batch, images_dict)
        self._get_fg_seg_rendering(outputs, batch, images_dict)
        self._get_ray_tracing_mask(outputs, batch, images_dict)
        self._get_ray_tracing_depth(outputs, batch, images_dict)

        return metrics_dict, images_dict

    def query_seg_field(self, inputs: TensorType["bs": ..., 3]) -> TensorType["bs": ..., 1]:
        """query segmentation field"""
        feat = self.feature_field.query_features(inputs, None)
        seg = self.feature_seg_field.compute_seg_prob(inputs, feat)
        return seg
    
    def seg_aware_sdf(self, inputs: TensorType["bs": ..., 3]) -> TensorType["bs": ..., 1]:
        """extract segmentation-awared sdf values"""
        sdf = self.field.forward_geonetwork(inputs)[..., [0]]
        seg = self.query_seg_field(inputs)
        
        # rectify sdf with seg
        rectified_sdf = self._rectify_sdf(sdf, seg)
        return rectified_sdf
    
    def _rectify_sdf(self, sdf: TensorType["bs": ..., 1], seg: TensorType["bs": ..., 1]):
        binary_seg = (seg > self.config.fg_3d_seg_binary_thr)[..., 0]
        bg_sdf = sdf[~binary_seg]
        min_bg_udf = self.config.sdf_rectification_value
        # sdf[~binary_seg] = torch.sign(bg_sdf) * torch.clamp(bg_sdf.abs(), min=min_bg_udf)  # Bad results, the ground plane is kept
        sdf[~binary_seg] = torch.clamp(bg_sdf.abs(), min=min_bg_udf)  # This might cause noises inside the fg region
        # TODO: visualize raw sdf (sdf viewer)
        return sdf
    
    def _get_pca_feature_rendering(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
        images_dict: Dict[str, torch.Tensor]
    ):
        feature = outputs["feature"]  # (h, w, c)
        feature_gt = batch["feature"]  # (c, h_gt, w_gt) - low res
        feature_gt_pca = batch["feature_pca"]  # (h_gt, w_gt, 3)
        all_gt_pca: PCA = batch["gt_features_pca"]
        h, w = feature.shape[:2]
        h_gt, w_gt = feature.shape[1:]
        min_val, max_val = batch["feature_pca_min"], batch["feature_pca_max"]

        # joint visualization of pred & gt (use PCA params fitted on gt features from all frames)
        feature_pca = all_gt_pca.transform(rearrange(feature.cpu().numpy(), 'h w c -> (h w) c'))
        feature_pca = rearrange(feature.new_tensor(feature_pca), '(h w) c -> h w c', h=h, w=w)
        feature_gt_pca = F.interpolate(
            feature_gt_pca.permute(2, 0, 1)[None], size=(h, w), mode="bilinear"
        )[0].permute(1, 2, 0)  # (h, w, 3)

        feature_pca_gt_pred = torch.cat([feature_gt_pca, feature_pca], dim=1)
        feature_pca_gt_pred = (feature_pca_gt_pred - min_val) / (max_val - min_val)

        images_dict["feature_pca_gt_pred"] = feature_pca_gt_pred
        # TODO: separate visualization of pred (use PCA params fitted on pred only)
        
        # pre-exp: compute a segmentation mask with feature_gt and self.ptcd_data
        # self._get_pseudo_gt_seg_mask(batch["image"].to(self.device), feature_gt, images_dict)

    def _get_fg_seg_rendering(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
        images_dict: Dict[str, torch.Tensor]
    ):
        # plot soft fg seg
        fg_seg_prob = outputs['fg_seg_prob']  # (h, w, 1)
        soft_fg_seg = apply_colormap(fg_seg_prob, cmap="viridis")
        
        # plot binary fg seg
        fg_seg_binary = (fg_seg_prob >= self.config.fg_seg_binary_thr).float()
        binary_fg_seg = fg_seg_binary.repeat(1, 1, 3)
        
        # overlay fg seg rendering on gt image
        image = batch["image"].to(self.device)
        # fg_seg_overlay = overlay_mask_on_image(soft_fg_seg, image, fg_seg_prob)
        fg_seg_overlay = overlay_mask_on_image(binary_fg_seg, image, fg_seg_prob)

        images_dict.update({
            "fg_seg_prob": soft_fg_seg,
            "fg_seg": binary_fg_seg,
            "fg_seg_overlay": fg_seg_overlay
        })

    def _get_pseudo_gt_seg_mask(self,
                                image_gt: TensorType["h", "w", 3],
                                feature_gt: TensorType["c", "h_gt", "w_gt"],
                                images_dict: Dict[str, torch.Tensor]):
        nc_feat, h_feat, w_feat = feature_gt.shape
        h, w = image_gt.shape[:2]
        
        feature_gt = rearrange(feature_gt, 'c h w -> (h w) c')
        assert not self.feature_seg_field.config.distance_weighted_average
        soft_psuedo_gt_mask = rearrange(
            self.feature_seg_field.compute_seg_prob(None, feature_gt),
            '(h w) 1 -> h w 1', h=h_feat, w=w_feat
        )
        soft_psuedo_gt_mask_hr = F.interpolate(
            soft_psuedo_gt_mask.permute(2, 0, 1)[None], size=(h, w), mode="bilinear"
        )[0].permute(1, 2, 0)  # (h, w, 1)
        binary_pseudo_gt_mask_hr = (soft_psuedo_gt_mask_hr > self.config.fg_seg_binary_thr).float()
        seg_overlay = overlay_mask_on_image(binary_pseudo_gt_mask_hr, image_gt, soft_psuedo_gt_mask_hr)
        images_dict["pseudo_gt_seg_overlay"] = seg_overlay

    def _get_ray_tracing_mask(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
        images_dict: Dict[str, torch.Tensor]
    ) -> None:
        if not self.config.use_surface_rendering:
            return
        
        image = batch["image"].to(self.device)
        
        # postprocessed non_convergent_mask
        convergent_masks = (~outputs["non_convergent_mask_rt"]).float()  # (h, w, 1)
        binary_rt_mask = convergent_masks.repeat(1, 1, 3)
        rt_mask_overlay = overlay_mask_on_image(binary_rt_mask, image, convergent_masks)
        
        # raw non_convergent_mask
        non_convergent_masks_raw = (~outputs["non_convergent_mask_rt_raw"]).float()  # (h, w, 1)
        binary_rt_mask = non_convergent_masks_raw.repeat(1, 1, 3)
        rt_mask_raw_overlay = overlay_mask_on_image(binary_rt_mask, image, non_convergent_masks_raw)
        
        # update images_dict
        images_dict.update({
            "rt_convergence_overlay": rt_mask_overlay,
            "rt_convergence_raw_overlay": rt_mask_raw_overlay,
        })
    
    def _get_ray_tracing_depth(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
        images_dict: Dict[str, torch.Tensor]
    ) -> None:
        if not self.config.use_surface_rendering:
            return
        
        depth = outputs["depth_rt"]
        convergent_mask = (~outputs["non_convergent_mask_rt"])
        min_depth = depth[convergent_mask].min()
        depth[~convergent_mask] = min_depth
        
        depth = apply_depth_colormap(depth, convergent_mask.float())
        images_dict["depth_rt"] = depth
    
    @torch.no_grad()
    def get_outputs_for_mesh_culling(
        self,
        ray_bundle: RayBundle,
        ray_sampler: Optional[Sampler] = None
    ) -> Dict[str, torch.Tensor]:
        """Render segmentation with foreground model for mesh culling."""
        self.handle_no_intersect_rays = False
        
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        num_rays = len(ray_bundle)
        outputs_lists = defaultdict(list)
        
        # chunked inference
        for i in tqdm(range(0, num_rays, num_rays_per_chunk), desc="Chunked inference for mesh culling"):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle_chunk = ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            
            outputs = self.render_fg_seg_for_mesh_culling(ray_bundle_chunk, ray_sampler=ray_sampler)
            
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        
        # merge chunks
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of non-tensors as well
                continue
            outputs[output_name] = torch.cat(outputs_list)
        return outputs

    @torch.no_grad()
    def render_fg_seg_for_mesh_culling(
        self,
        ray_bundle: RayBundle,
        ray_sampler: Optional[Sampler] = None
    ) -> Dict[str, torch.Tensor]:
        # query seg_probs & weights
        if ray_sampler is None:  # use the same sampler as for training
            samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)
            _weights_key = "weights" if not self.config.use_surface_rendering else "weights_rt"
            weights = samples_and_field_outputs[_weights_key].detach()
            fg_seg_probs = samples_and_field_outputs[FieldHeadNames.FG_SEG].detach()
        else:
            if self.config.use_surface_rendering:
                assert isinstance(ray_sampler, DummySampler)
                ray_samples = ray_sampler(ray_bundle)
                weights = torch.ones_like(ray_samples.frustums.starts)
            else:
                assert isinstance(ray_sampler, SpacedSampler)
                ray_samples = ray_sampler(ray_bundle, num_samples=ray_sampler.num_samples)
                field_outputs = self.field(ray_samples, return_alphas=True)
                weights = ray_samples.get_weights_from_alphas(field_outputs[FieldHeadNames.ALPHA])
            feature_field_outputs = self.feature_field(ray_samples)
            feature_field_seg_outptus = self.feature_seg_field(
                ray_samples, feature_field_outputs[FieldHeadNames.FEATURE]
            )
            fg_seg_probs = feature_field_seg_outptus[FieldHeadNames.FG_SEG].detach()
        
        # rendering
        fg_seg_prob_rendering = self.renderer_fg_seg(fg_seg_probs, weights=weights.detach())
        outputs = {
            "fg_seg_prob": fg_seg_prob_rendering,
            "fg_seg_probs": fg_seg_probs,
            "weights": weights
        }
        return outputs
