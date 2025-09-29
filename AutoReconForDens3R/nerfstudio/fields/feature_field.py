from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Type

from rich.console import Console
from torchtyping import TensorType
from typing_extensions import Literal

try:
    import faiss
except ImportError as err:
    pass
import numpy as np
import torch
from torch.nn import Parameter
from torch.nn import functional as F

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, FieldConfig
from nerfstudio.utils.pointclouds import BasicPointClouds

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass

CONSOLE = Console(width=128)


@dataclass
class FeatureFieldConfig(FieldConfig):
    """Feature Field Config"""

    _target: Type = field(default_factory=lambda: FeatureField)

    nc_position: int = 3
    """Number of channels for positional inputs"""
    nc_feature: int = 384
    nc_hidden: int = 128
    n_hidden_layers: int = 2
    use_viewdirs: bool = False
    use_grid_feature: bool = True
    """Whether to use multi-resolution feature grids (only support hash encoding for now)"""
    hash_grid_max_res: int = 128
    """Maximum resolution of the feature grid (use a smaller default value since ViT features is of lower frequency)"""
    hash_grid_log2_hashmap_size: int = 19
    # hash_grid_precision: Literal["float32", "float16"] = "float32"
    use_position_encoding: bool = True
    """Whether to include positional encodings beyond the grid features (for better smoothness)"""
    use_fused_encoding_mlp: bool = True
    """Whether to use TCNN fused encoding & mlp"""


class FeatureField(Field):
    """A indivisual feature field without feature sharing with the main field."""

    config: FeatureFieldConfig

    def __init__(self, config: FeatureFieldConfig, spatial_distortion: Optional[SpatialDistortion] = None) -> None:
        super().__init__()
        self.config = config

        self.spatial_distortion = spatial_distortion
        self.use_grid_feature = self.config.use_grid_feature
        self.use_viewdirs = self.config.use_viewdirs

        # feature field hash grid
        if self.config.use_viewdirs:
            raise NotImplementedError

        # if self.config.use_position_encoding:
        #     raise NotImplementedError  # tcnn nested positional encoding not working

        if self.config.use_grid_feature:
            self.build_ngp()
        else:
            self.build()

    def build_ngp(self):
        L, F, N_min = 16, 2, 16
        log2_T = self.config.hash_grid_log2_hashmap_size
        N_max = self.config.hash_grid_max_res
        b = np.exp(np.log(N_max / N_min) / (L - 1))  # feature is of lower-frequency -> N_max=128
        nc_pos = self.config.nc_position

        hash_encoding_config = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": L,
            "n_features_per_level": F,
            "log2_hashmap_size": log2_T,
            "base_resolution": N_min,
            "per_level_scale": b,
            "n_dims_to_encode": nc_pos,
            "interpolation": "Linear",  # TODO: support "Smoothstep"
        }

        if self.config.use_fused_encoding_mlp:  # tcnn encoding + mlp
            encoding_config = {
                "otype": "Composite",
                # Hash for the first nc_pos dims; Frequency for the later nc_pos dims
                "nested": [
                    hash_encoding_config,
                    {
                        "otype": "Frequency",
                        "n_frequencies": 8,
                        "n_dims_to_encode": nc_pos,
                    },
                ],
            }

            # when tinycudann is built w/ float16 support, it will use float16 automatically
            self.feature_encoder = tcnn.NetworkWithInputEncoding(
                n_input_dims=nc_pos * 2
                if self.config.use_position_encoding
                else nc_pos,  # 2 * nc_pos for smooth encoding
                n_output_dims=self.config.nc_feature,
                encoding_config=encoding_config,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.config.nc_hidden,
                    "n_hidden_layers": self.config.n_hidden_layers,
                },
                # dtype=torch.float32 if self.config.hash_grid_precision == "float32" else torch.float16
            )
        else:  # tcnn encoding + torch MLP
            raise NotImplementedError
            # self.pos_encoder = tcnn.Encoding(n_input_dims=nc_pos,
            #                                  encoding_config=hash_encoding_config)
            # pos_dim = self.pos_encoder.n_output_dims

            # self.mlp = nn.Sequential(
            #     fc_block(pos_dim, self.config.nc_hidden),
            #     *[fc_block(self.config.nc_hidden, self.config.nc_hidden) for _ in range(self.config.n_hidden_layers)],
            #     nn.Linear(self.config.nc_hidden, self.config.nc_feature)  # TODO: support output activation
            # )

    def build(self):
        raise NotImplementedError

    def query_features(
        self, positions: TensorType["n_rays", "d_pos"], viewdirs: Optional[TensorType["n_rays", "d_dir"]] = None
    ) -> TensorType["n_rays", "d_feature"]:
        """Qeury the feature field"""
        x = self._get_input_positions(positions)

        if self.config.use_position_encoding:
            # FIXME: raise runtime_error after loading checkpoints to other fields.
            features = self.feature_encoder(torch.cat([x, x], dim=-1))
        else:
            features = self.feature_encoder(x)
        return features

    def get_outputs(self, ray_samples: RaySamples) -> Dict[str, torch.Tensor]:
        outputs = {}

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)
        features = self.query_features(inputs, directions_flat)  # NOTE: feautures.dtype = torch.float16

        features = features.view(*ray_samples.frustums.directions.shape[:-1], -1)

        outputs = {FieldHeadNames.FEATURE: features}
        return outputs

    def forward(self, ray_samples: RaySamples):
        """Evaluates the field at the given ray samples"""
        field_outputs = self.get_outputs(ray_samples)
        return field_outputs

    def _get_input_positions(self, positions: torch.Tensor):
        """compute input positions (for positional encoding / feature encoding)
        from raw ray sample positions.

        NOTE: assume the whole scene is represented by a single feature field
        """
        positions = self.spatial_distortion(positions)  # [[-2, -1], [-1, 1], [1, 2]]
        positions = (positions + 2.0) / 4.0  # Full normalization: [-2, 2] -> [0, 1]
        # FIXME: handle positions out of [0, 1] after spatial_distortion & normalization
        return positions


@dataclass
class FeatureSegFieldConfig(FieldConfig):
    """FeatureSegField Config"""

    _target: Type = field(default_factory=lambda: FeatureSegField)

    knn: int = 10  # TOOD: seg fg & bg knn separately?
    """number of reference points to retrieve for each point to be segmented"""
    distance_weighted_average: bool = False
    """compute distance-weighted average of cosine similairties of the knn retrieved points"""
    in_fg_aabb_only: bool = True
    """if a point is not in the fg aabb, it is directly considered as background"""
    contrast_fg_bg: bool = True
    """whether to derive segmentation field by contrast between foreground and background points"""
    separate_bg_plane: bool = False
    """when computing segmentation probs, treat background and plane as two different classes"""


class FeatureSegField(Field):
    """calculate segmentation field from feature field"""

    config: FeatureSegFieldConfig

    def __init__(self, config: FeatureFieldConfig, ptcd_data: BasicPointClouds, aabb: TensorType[2, 3]) -> None:
        super().__init__()
        self.config = config
        self.ptcd_data = ptcd_data
        self.aabb = Parameter(aabb, requires_grad=False)
        self.ref_feats_fg = ptcd_data.fg_feats
        self.ref_feats_bg = torch.cat([ptcd_data.bg_feats, ptcd_data.plane_feats], 0)
        self.ref_pts_fg = ptcd_data.fg_pts
        self.ref_pts_bg = torch.cat([ptcd_data.bg_pts, ptcd_data.plane_pts], 0)
        self.fg_nn_index = ptcd_data.fg_nn_index
        self.bg_nn_index = ptcd_data.bg_nn_index

        if self.config.separate_bg_plane:
            self.plane_nn_index = ptcd_data.plane_nn_index
            self.ref_feats_bg, self.ref_pts_bg = ptcd_data.bg_feats, ptcd_data.bg_pts
            self.ref_feats_plane, self.ref_pts_plane = ptcd_data.plane_feats, ptcd_data.plane_pts

        self._register_all_tensors_as_buffers()

    @torch.no_grad()
    def compute_seg_prob(
        self,
        points: Optional[TensorType["num_samples", 3]],  # unnormalized, raw positions
        features: TensorType["num_samples", "nc_features"],
        ptcd_data: Optional[BasicPointClouds] = None,
    ) -> TensorType["num_samples", 1]:
        if ptcd_data is not None:  # Override self.ptcd_data
            raise NotImplementedError

        features = features.detach()
        if points is not None:
            points = points.detach()
        if self.config.in_fg_aabb_only:
            assert points is not None
            in_fg_aabb_mask = ((points >= self.aabb[0]) & (points <= self.aabb[1])).all(-1)  # (n, )
            # TODO: for pts not in fg aabb, omit the seg_probs computation directly.

        feats_l2_normalized = F.normalize(features, p=2, dim=-1)

        knn_inds_fg, knn_feats_fg, knn_pts_fg = self.search_knn_fg(feats_l2_normalized)
        knn_cos_sims_avg_fg = self.compute_cos_sim(points, feats_l2_normalized, knn_feats_fg, knn_pts_fg)

        if not self.config.contrast_fg_bg:
            seg_probs = knn_cos_sims_avg_fg * 0.5 + 0.5
        else:
            knn_inds_bg, knn_feats_bg, knn_pts_bg = self.search_knn_bg(features)
            knn_cos_sims_avg_bg = self.compute_cos_sim(points, feats_l2_normalized, knn_feats_bg, knn_pts_bg)

            # contrastive seg_probs
            if self.config.separate_bg_plane:
                knn_inds_plane, knn_feats_plane, knn_pts_plane = self.search_knn_plane(features)
                knn_cos_sims_avg_plane = self.compute_cos_sim(
                    points, feats_l2_normalized, knn_feats_plane, knn_pts_plane
                )  # (n, 1)
                knn_cos_sims_avg_bg = torch.maximum(knn_cos_sims_avg_bg, knn_cos_sims_avg_plane)

            # TODO: tune temperature and offset cos_sims before applying softmax (since most cos_sims > -0.1)
            seg_probs = torch.softmax(torch.cat([knn_cos_sims_avg_fg, knn_cos_sims_avg_bg], -1), dim=-1)
            seg_probs = seg_probs[..., [0]]

        if self.config.in_fg_aabb_only:
            seg_probs[~in_fg_aabb_mask] = 0.0
        return seg_probs

    def search_knn_fg(self, feats: TensorType["n", "c"]):
        return self._search_knn(feats, self.ref_feats_fg, self.ref_pts_fg, self.fg_nn_index)

    def search_knn_bg(self, feats: TensorType["n", "c"]):
        return self._search_knn(feats, self.ref_feats_bg, self.ref_pts_bg, self.bg_nn_index)

    def search_knn_plane(self, feats: TensorType["n", "c"]):
        return self._search_knn(feats, self.ref_feats_plane, self.ref_pts_plane, self.plane_nn_index)

    def _search_knn(
        self,
        feats: TensorType["n", "c"],
        ref_feats: TensorType["m", "c"],
        ref_pts: TensorType["m", 3],
        nn_index: faiss.Index,
    ) -> Tuple[TensorType["n", "k"], TensorType["n", "k", "c"], TensorType["n", "k", 3]]:
        feats = feats.to(torch.float32)
        knn_dists, knn_inds = nn_index.search(feats, self.config.knn)  # (n, k)
        knn_feats = torch.index_select(ref_feats, 0, knn_inds.reshape(-1)).reshape(*knn_inds.shape, -1)  # (n, k, c)
        knn_pts = torch.index_select(ref_pts, 0, knn_inds.reshape(-1)).reshape(*knn_inds.shape, -1)  # (n, k, c)
        return knn_inds, knn_feats, knn_pts

    def compute_cos_sim(
        self,
        points: Optional[TensorType["n", 3]],
        feats: TensorType["n", "c"],
        knn_feats: TensorType["n", "k", "c"],
        knn_pts: TensorType["n", "k", 3],
    ) -> TensorType["n", 1]:
        knn_cos_sims = F.cosine_similarity(feats[:, None], knn_feats, dim=-1)  # (n, k)
        if not self.config.distance_weighted_average:
            return knn_cos_sims.mean(-1, keepdim=True)

        knn_pts_dists = torch.linalg.norm(points[:, None] - knn_pts, ord=2, dim=-1)
        knn_pts_weights = torch.exp(-knn_pts_dists)  # TODO: tune temperature
        knn_pts_weights = knn_pts_weights / knn_pts_weights.sum(-1, keepdim=True)  # normalized weights
        # use unnormalized weights would down weight bg_cos_sims too much
        knn_weighed_cos_sims = (knn_cos_sims * knn_pts_weights).sum(-1, keepdim=True)
        return knn_weighed_cos_sims

    def _register_all_tensors_as_buffers(self):
        """register all Tensor attributes as buffers to automatically move them to the target device"""
        _attr_names = list(vars(self).keys())
        for _name in _attr_names:
            _attr = getattr(self, _name)
            if not isinstance(_attr, torch.Tensor):
                continue

            delattr(self, _name)
            self.register_buffer(_name, _attr, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        ray_samples: Optional[RaySamples],
        features: TensorType["bs":..., "num_samples", "nc_features"],
        ptcd_data: Optional[BasicPointClouds] = None,
    ) -> Dict[str, TensorType["bs":..., 1]]:
        """Calculate segmentation along the ray."""
        prefix_dims, nc_feat = features.shape[:-1], features.shape[-1]
        features = features.view(-1, nc_feat)

        raw_positions = ray_samples.frustums.get_start_positions().view(-1, 3) if ray_samples is not None else None

        seg_probs = self.compute_seg_prob(raw_positions, features, ptcd_data=ptcd_data).reshape(*prefix_dims, 1)

        outputs = {FieldHeadNames.FG_SEG: seg_probs}
        return outputs
