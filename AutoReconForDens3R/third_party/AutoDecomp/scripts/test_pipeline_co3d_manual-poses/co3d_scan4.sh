
INST_REL_DIR=CO3D_DEMO/scan4
FORCE_RERUN=True

python auto_decomp/cli/inference_transformer.py --config-name=config \
    inst_rel_dir=$INST_REL_DIR \
    sparse_recon.n_images=40 \
    sparse_recon.force_rerun=$FORCE_RERUN \
    sparse_recon.n_feature_workers=1 sparse_recon.n_recon_workers=1 \
    triangulation.force_rerun=$FORCE_RERUN \
    triangulation.n_feature_workers=1 triangulation.n_recon_workers=1 \
    dino_feature.force_extract=$FORCE_RERUN dino_feature.n_workers=1
