
DATA_ROOT=assets/
INST_REL_DIR=custom_data_example/co3d_chair
FORCE_RERUN=True

python auto_decomp/cli/inference_transformer.py --config-name=config \
    data_root=$DATA_ROOT \
    inst_rel_dir=$INST_REL_DIR \
    sparse_recon.n_images=40 \
    sparse_recon.force_rerun=$FORCE_RERUN \
    sparse_recon.n_feature_workers=1 sparse_recon.n_recon_workers=1 \
    triangulation.force_rerun=$FORCE_RERUN \
    triangulation.n_feature_workers=1 triangulation.n_recon_workers=1 \
    dino_feature.force_extract=$FORCE_RERUN dino_feature.n_workers=1
