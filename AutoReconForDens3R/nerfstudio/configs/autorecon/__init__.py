from __future__ import annotations

from typing import Dict, Sequence
from functools import partial

import tyro
from sklearn.decomposition import PCA

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import (
    Config,
    SchedulerConfig,
    TrainerConfig,
    ViewerConfig,
)
from nerfstudio.data.datamanagers.base_datamanager import (
    FlexibleDataManagerConfig,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datamanagers.semantic_datamanager import SemanticDataManagerConfig
from nerfstudio.data.datamanagers.autorecon_datamanager import AutoReconDataManagerConfig
from nerfstudio.data.datamanagers.variable_res_datamanager import (
    VariableResDataManagerConfig,
)
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.friends_dataparser import FriendsDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import (
    PhototourismDataParserConfig,
)
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.data.dataparsers.autorecon_dataparser import AutoReconDataParserConfig
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig, AdamWOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialSchedulerConfig,
    MultiStepSchedulerConfig,
    NeuSSchedulerConfig,
    AutoReconSchedulerConfig
)

from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.fields.feature_field import FeatureFieldConfig, FeatureSegFieldConfig
from nerfstudio.models.dto import DtoOModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.neuralreconW import NeuralReconWModelConfig
from nerfstudio.models.neus import NeuSModelConfig
from nerfstudio.models.neus_acc import NeuSAccModelConfig
from nerfstudio.models.neus_facto import NeuSFactoModelConfig
from nerfstudio.models.neus_facto_dff import NeuSFactoDFFModelConfig
from nerfstudio.models.neus_facto_reg import NeuSFactoRegModelConfig
from nerfstudio.models.distilled_neus_facto import DistilledNeuSFactoModelConfig
from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.models.unisurf import UniSurfModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.models.volsdf import VolSDFModelConfig
from nerfstudio.pipelines.base_pipeline import (
    FlexibleInputPipelineConfig,
    VanillaPipelineConfig,
)
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.utils.func_utils import get_first_element
from nerfstudio.utils.pointclouds import BasicPointClouds

from .baseline import neus_wbg, neus_wbg_mlp
from .common import (
    neusfacto_autorecon, neus_facto_wbg, neus_facto_wbg_BgProbNet,
    neus_facto_wbg_medium, neus_facto_wbg_tiny, neus_facto_wbg_tiny_long_schedule
)
from .distilled_neusfacto import distilled_neus_facto_wbg, debug_distilled_neus_facto_wbg
from .feature_field import neus_facto_wbg_fast_dff, neus_facto_dff_wbg
from .neusfacto_fast import neus_facto_wbg_fast
from .regularization import neus_facto_wbg_reg, neus_facto_wbg_reg_sep_plane, neus_facto_wbg_reg_sep_plane_nerf
from .semantic_nerf import autorecon_semantic_nerf


method_configs: Dict[str, Config] = {
    "neus-wbg": neus_wbg,
    "neus-wbg_mlp": neus_wbg_mlp,
    "neus-facto-autorecon": neusfacto_autorecon,
    "neus-facto-wbg": neus_facto_wbg,
    "neus-facto-wbg_bg-prob-net": neus_facto_wbg_BgProbNet,
    "neus-facto-wbg_medium": neus_facto_wbg_medium,
    "neus-facto-wbg_tiny": neus_facto_wbg_tiny,
    "neus-facto-wbg_tiny_long-schedule": neus_facto_wbg_tiny_long_schedule,
    "distilled-neus-facto-wbg": distilled_neus_facto_wbg,
    "debug_distilled-neus-facto-wbg": debug_distilled_neus_facto_wbg,
    "neus-facto-wbg-fast_dff": neus_facto_wbg_fast_dff,
    "neus-facto-dff-wbg": neus_facto_dff_wbg,
    "neus-facto-wbg-fast": neus_facto_wbg_fast,
    "neus-facto-wbg-reg": neus_facto_wbg_reg,
    "neus-facto-wbg-reg_sep-plane": neus_facto_wbg_reg_sep_plane,
    "neus-facto-wbg-reg_sep-plane-nerf": neus_facto_wbg_reg_sep_plane_nerf,
    "autorecon_semantic-nerfw": autorecon_semantic_nerf
}
