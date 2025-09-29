from hydra_zen import builds, store

from .point_transformer import PointTransformer, TokenSegHead

TransformerConfig = builds(PointTransformer, populate_full_signature=True)
base_transformer = TransformerConfig(
    depth=2, embed_dim=96, linear_attention=True, pos_enc_type="discrete_sine", pos_enc_merge="sum"
)

transformer_backbone_store = store(group="decomp/transformer/backbone")
transformer_backbone_store(base_transformer, name="base")

SegHeadConfig = builds(TokenSegHead, populate_full_signature=True)
base_seg_head = SegHeadConfig(use_softmax=False, num_classes=2, temperature=1.0)
seg_head_store = store(group="decomp/transformer/seg_head")
seg_head_store(base_seg_head, name="base")

store.add_to_hydra_store()
