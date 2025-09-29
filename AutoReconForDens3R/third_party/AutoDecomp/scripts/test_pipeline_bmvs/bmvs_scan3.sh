
INST_REL_DIR=BlendedMVS/scan3
FORCE_RERUN=True

python auto_decomp/cli/inference_transformer.py --config-name=idr \
    inst_rel_dir=$INST_REL_DIR \
    triangulation.force_rerun=$FORCE_RERUN \
    triangulation.n_feature_workers=1 triangulation.n_recon_workers=1 \
    dino_feature.force_extract=True dino_feature.n_workers=1
