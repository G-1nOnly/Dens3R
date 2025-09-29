"""
(sfm -> extract dino features) -> preprocess sfm -> decompose w/ transformer
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import MISSING
from loguru import logger
from omegaconf import OmegaConf

import os
import sys
# sys.path.append("/home/meta3d/Desktop/code_4070ti/dust3r/AutoRecon/third_party/AutoDecomp")
sys.path.append(str(Path(os.path.abspath(__file__)).resolve().parents[2]))

from auto_decomp.decomp.preprocess import SfMPreprocess, SfMPreprocessConfig
from auto_decomp.decomp.transformer.dataset.utils import build_inference_batch
from auto_decomp.decomp.transformer.lightning.module import (
    SfMTransformer,
    SfMTransformerConfig,
)
from auto_decomp.feature_extraction.dino_vit import extract_features as dino_extractor
from auto_decomp.sfm import colmap_from_idr, sfm

SfmMode = Enum("SfmMode", ["sparse_recon", "idr2colmap", "dens3r"])


@dataclass
class Config:
    defaults: List[Any] = field(
        default_factory=lambda: [
            {"sparse_recon": "sequential"},
            {"idr2colmap": "base"},
            {"triangulation": "sequential"},
            {"dino_feature": "base"},
            {"decomp/sfm_preprocess": "base"},
            {"decomp/transformer": "base"},
        ]
    )

    data_root: Path = Path("data/")
    inst_rel_dir: str = MISSING
    sfm_mode: SfmMode = SfmMode.sparse_recon
    sparse_recon: sfm.SfMConfig = MISSING
    idr2colmap: colmap_from_idr.IDR2COLMAPConfig = MISSING
    triangulation: sfm.SfMConfig = MISSING
    dino_feature: dino_extractor.DINOExtractionConfig = MISSING
    ckpt_path: str = "ckpts/no-chair.ckpt"
    save_dirname: str = "auto-decomp_transformer"


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


def _update_config(cfg):
    inst_rel_dir = cfg.inst_rel_dir
    for k in ["sparse_recon", "idr2colmap", "triangulation", "dino_feature", "decomp.sfm_preprocess"]:
        OmegaConf.update(cfg, f"{k}.data_root", cfg.data_root)
        OmegaConf.update(cfg, f"{k}.inst_rel_dir", inst_rel_dir)


def main(config: Config):
    cfg = config
    _update_config(cfg)

    sfm_preprocess = SfMPreprocess(cfg.decomp.sfm_preprocess)
    sfm_transformer = SfMTransformer(cfg.decomp.transformer)

    if Path(cfg.ckpt_path).exists():
        ckpt_path = cfg.ckpt_path
    else:
        ckpt_path = Path(__file__).parents[2] / cfg.ckpt_path
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    sfm_transformer.load_state_dict(state_dict, strict=True)
    sfm_transformer.to("cuda")

    if cfg.sfm_mode == SfmMode.sparse_recon:
        logger.info("Running sfm w/ sparse features")
        sfm.main(cfg.sparse_recon)
        logger.info("Triangulating LoFTR matches")
        sfm.main(cfg.triangulation)
    elif cfg.sfm_mode == SfmMode.idr2colmap:
        logger.info("Converting annotation in the IDR format to a COLMAP model")
        colmap_from_idr.main(cfg.idr2colmap)
        logger.info("Triangulating LoFTR matches")
        sfm.main(cfg.triangulation)
    elif cfg.sfm_mode == SfmMode.dens3r:
        logger.info("Running sfm w/ dens3r")
        sfm_dir_name = sfm.dens3r_recon(cfg.sparse_recon)
        sfm.postprocess_dens3r(cfg.triangulation, sfm_dir_name)
    else:
        raise ValueError(f"Unknown sfm mode: {cfg.sfm_mode}")

    logger.info("Extracting DINO features")
    dino_extractor.main(cfg.dino_feature)

    logger.info("Preprocessing LoFTR triangulation")
    data: Dict[str, Any] = sfm_preprocess()

    logger.info("Decomposing sfm point cloud w/ sfm_transformer")
    sfm_preprocess_cfg = cfg.decomp.sfm_preprocess
    inst_dir = sfm_preprocess_cfg.data_root / sfm_preprocess_cfg.inst_rel_dir
    data["inst_dir"] = inst_dir
    batch = build_inference_batch(data, facet="key", device="cuda")
    sfm_transformer.inference(batch)
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    hydra.main(config_name="base_config", config_path="../configs/inference_transformer", version_base=None)(main)()
