
scan_name=scan3

DATA_ROOT=data
INST_REL_DIR=CO3D_DEMO/$scan_name
FORCE_RERUN=True

python third_party/AutoDecomp/auto_decomp/cli/inference_transformer.py --config-name=cvpr \
    data_root=$DATA_ROOT \
    inst_rel_dir=$INST_REL_DIR \
    sparse_recon.n_images=40 \
    sparse_recon.force_rerun=$FORCE_RERUN \
    sparse_recon.n_feature_workers=1 sparse_recon.n_recon_workers=1 \
    triangulation.force_rerun=$FORCE_RERUN \
    triangulation.n_feature_workers=1 triangulation.n_recon_workers=1 \
    dino_feature.force_extract=$FORCE_RERUN dino_feature.n_workers=1 

ns-train neus-facto-wbg-reg_sep-plane-nerf \
    --experiment-name neusfacto-wbg-reg_sep-plane-nerf_60k_plane-h-ratio-0.3_co3d-${scan_name}_cvpr \
    --vis tensorboard \
    --trainer.steps_per_eval_image 2500 \
    --trainer.steps_per_eval_batch 2500 \
    --trainer.max_num_iterations 60001 \
    --trainer.steps_per_save 60000 \
    --pipeline.model.cos_anneal_end 10000 \
    --pipeline.model.plane_height_ratio 0.3 \
    --optimizers.fields.scheduler.max_steps 60001 \
    --pipeline.datamanager.eval_camera_res_scale_factor 0.25 \
    --pipeline.model.sdf_field.hash_grid_progressive_training False \
    autorecon-data --data $DATA_ROOT/$INST_REL_DIR \
    --anno_dirname 'triangulate_loftr-720000_sequential_np-10/auto-deocomp_sfm-transformer_cvpr' \
    --camera_filename cameras_cameras_norm-obj-side-2.0.npz \
    --object_filename objects_cameras_norm-obj-side-2.0.npz \
    --parse_images_from_camera_dict True \
    --image_dirname "images" --image_extension ".jpg" \
    --include_image_features False --include_coarse_features False \
    --use_accurate_scene_box True \
    --collider_type 'box_near_far' \
    --near_far 0.1 100.0 \
    --compute_fg_bbox_mask True \
    --force_recompute_fg_bbox_mask False \
    --decomposition_mode regularization \
    --downsample_ptcd -1

# LOG_DIR="outputs/<path_to_log_dir>"

# MC_RES=512  # 1024 (default)
# MESH_FN="extracted_mesh_res-${MC_RES}.ply"
# MESH_PATH="${LOG_DIR}/${MESH_FN}"

# ns-extract-mesh \
# 	--load-config $LOG_DIR/config.yml \
#     --load-dir $LOG_DIR/sdfstudio_models \
# 	--output-path $MESH_PATH \
#     --chunk_size 25000 --store_float16 True \
#     --resolution $MC_RES \
#     --use_train_scene_box True \
#     --seg_aware_sdf False \
#     --remove_internal_geometry None \
#     --remove_non_maximum_connected_components True \
#     --close_holes False --simplify_mesh_final False
