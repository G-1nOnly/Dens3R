from dataclasses import dataclass, field
from typing import Any, List, Literal

import pytorch_lightning as pl
from hydra_zen import MISSING, instantiate, make_custom_builds_fn, store
from hydra_zen.typing import Partial
from loguru import logger
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import MultiStepLR

from auto_decomp.decomp.transformer.modeling import (
    PointTransformer,
    SegHeadConfig,
    TokenSegHead,
    TransformerConfig,
)
from auto_decomp.decomp.transformer.utils.postprocess import (
    Postprocess,
    PostprocessConfig,
)
from auto_decomp.decomp.transformer.utils.saving import (
    SaveResults,
    SfMTransformerSaveConfig,
)

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)


def if_none_then_true(x):
    return x is None


@dataclass
class SfMTransformerConfig:
    defaults: List[Any] = field(
        default_factory=lambda: [
            {"backbone": "base"},
            {"seg_head": "base"},
            {"postprocess_train": "base"},
            {"postprocess_test": "inference"},
            {"saving": "test_autorecon"},
        ]
    )
    # model
    backbone: TransformerConfig = MISSING
    seg_head: SegHeadConfig = MISSING
    # training
    optimizer: Partial[Optimizer] = pbuilds(Adam, lr=1e-3)
    scheduler: Partial[MultiStepLR] = pbuilds(MultiStepLR, milestones=[5, 10, 15, 20, 25], gamma=0.1)
    # postprocess & decomposition buiding
    postprocess_train: PostprocessConfig = MISSING
    postprocess_test: PostprocessConfig = MISSING  # either eval / inference
    # saving
    saving: SfMTransformerSaveConfig = MISSING

    # TODO: vis configs


sfm_trans_store = store(group="decomp/transformer")
sfm_trans_store(SfMTransformerConfig, name="base")
store.add_to_hydra_store()

class SfMTransformer(pl.LightningModule):
    def __init__(self, cfg: SfMTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer: PointTransformer = instantiate(cfg.backbone)
        self.seg_head: TokenSegHead = instantiate(cfg.seg_head)
        self.postprocess_train = Postprocess(cfg.postprocess_train)
        self.postprocess_test = Postprocess(cfg.postprocess_test)
        self.save_results = SaveResults(cfg.saving)

    def forward(self, data):
        xyz, feat, cls_feat = map(data.get, ["xyz", "feat", "cls_feat"])
        tokens = self.transformer(xyz, feat, cls_token=cls_feat)  # (B, 1+N, C)
        probs = self.seg_head(tokens[:, 1:], tokens[:, [0]])  # (B, N, 1)
        return probs

    def inference(self, batch):
        return self.test_val_step(batch, -1, mode="test")

    def test_val_step(self, batch, batch_idx, mode="test"):
        assert mode in ["val", "test"]  # TODO: check whether "val" is correctly handled
        # is_sanity_check = self.trainer.state.stage == "sanity_check"
        aux_data = self.build_aux_attrs(batch, mode)

        probs = self.forward(batch)
        fg_aabb_verts_no_postprocess, fg_aabb_verts, decomp_results = self.postprocess_test(
            batch["xyz"],
            probs,
            avg_cam_loc=batch.get("avg_cam_loc", None),
            aux_data=aux_data,
            rot_mat=batch.get("rot_mat", None),
            instance_names=batch["inst_name"],
        )

        # save annotations
        if mode == "test":
            self.save_results.save_annotations(batch, decomp_results)

        # TODO: compute losses & metrics
        # TODO: visualization (random, fixed)
        # TODO: return losses & metrics
        return {}

    def validation_step(self, batch, batch_idx):
        return self.test_val_step(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.test_val_step(batch, batch_idx, mode="test")

    def build_aux_attrs(self, batch, mode: Literal["train", "test", "val"]):
        """Build auxiliary point-wise attributs."""
        if mode == "test":
            # TODO: build a enum shared with the dataset class
            try:
                aux_keys = ["rgb_all", "pca_rgb_all", "feat"]  # NOTE: augmentations are not applied on these fields
                aux_data = {k: batch[k] for k in aux_keys}
            except KeyError:
                logger.warning("Missing auxiliary point-wise attributs, ignored!")
                aux_data = None
        else:
            aux_data = None
        return aux_data
