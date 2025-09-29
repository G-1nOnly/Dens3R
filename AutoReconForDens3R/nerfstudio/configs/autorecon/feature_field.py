from . import *


neus_facto_wbg_fast_dff = Config(
    method_name="neus-facto-wbg-fast_dff",
    trainer=TrainerConfig(
        steps_per_eval_image=1000,
        steps_per_eval_batch=1000,
        steps_per_save=5000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=5001,
        mixed_precision=True,
        load_pipeline_ckpt_only=True,
        load_pipeline_ckpt_strict=False
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=AutoReconDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
            eval_camera_res_scale_factor=1.0,
            collate_fn=partial(nerfstudio_collate,
                               extra_mappings={PCA: get_first_element,
                                               BasicPointClouds: get_first_element}),
            train_mask_sample_start=2000,  # TODO: can be more aggresive e.g., 1
            train_mask_sample_ratio=0.8  # quality of regions out of the fg bbox mask is bad, so learning good feature field on bad geometry is meaningless
        ),
        model=NeuSFactoDFFModelConfig(
            cos_anneal_end=0,  # must set to 0, as the sdf_field is already trained!
            use_proposal_weight_anneal=False,  # as the proposal net is already trained
            use_proposal_weight_anneal_bg=False,  # as the proposal net is already trained
            sdf_field=SDFFieldConfig(
                use_position_encoding=True,
                use_grid_feature=True,
                num_layers=2,
                num_layers_color=2,
                hidden_dim=256,
                bias=0.1,  # 0.5 -> 0.1
                beta_init=0.3,
                direction_encoding_type="sh",
                use_appearance_embedding=False,
                inside_outside=False,  # only consider object-centric scenes
                hash_grid_progressive_training=True,
                hash_grid_progressive_training_iters=-1,  # use weights=1
                spatial_normalization_region="aabb",  # to better utilize the feature grid
                weight_norm=False  # weight_norm might lead to NaNs during training
            ),
            handle_bottom_intersection_rays=True,  # this is vital to force the fg model to reconstruct the ground plane
            proposal_net_spatial_normalization_region="fg",
            eval_num_rays_per_chunk=1024,
            use_average_appearance_embedding=False,
            background_model="grid",
            bg_use_appearance_embedding=False,
            use_fg_aware_scene_contraction=True,
            fg_aware_scene_contraction_alpha=5.0,  # NOTE: the reconstruction result is sensitive to this parameter, better set adaptively according to the coarse sfm reconstruction
            bg_sampler_type="proposal_network",
            num_samples_outside=16,
            num_bg_proposal_samples_per_ray=(96, 64),
            use_surface_rendering=True,
            feature_field=FeatureFieldConfig(
                hash_grid_max_res=128,
                use_position_encoding=True,  # smooth encoding
                use_fused_encoding_mlp=True
            ),
            feature_field_scene_contraction_scale_factor=0.25,  # TODO: tune scale factor
            feature_loss_mult=1.0,
            render_fg_seg=True,
            fg_seg_binary_thr=0.5,
            feature_seg_field=FeatureSegFieldConfig(
                knn=10,
                contrast_fg_bg=True
            )
        ),
    ),
    optimizers={
        "feature_field": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=8000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)


# nesu-facto-wgb with Feature Field
neus_facto_dff_wbg = Config(
    method_name="neus-facto-dff-wbg",
    trainer=TrainerConfig(
        steps_per_eval_image=2500,
        steps_per_eval_batch=2500,
        steps_per_save=10000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=20001,
        mixed_precision=True,
        load_pipeline_ckpt_only=True,
        load_pipeline_ckpt_strict=False
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=AutoReconDataParserConfig(),  # NOTE: cannot set init args here, which would not be recognised.
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
            eval_camera_res_scale_factor=1.0,
            collate_fn=partial(nerfstudio_collate,
                               extra_mappings={PCA: get_first_element,
                                               BasicPointClouds: get_first_element})
        ),
        model=NeuSFactoDFFModelConfig(
            sdf_field=SDFFieldConfig(
                # TODO: use SH instead of sine
                use_position_encoding=False,
                use_grid_feature=True,
                hash_grid_precision="float16",
                num_layers=2,
                num_layers_color=2,
                hidden_dim=256,
                bias=0.5,
                beta_init=0.3,
                use_appearance_embedding=False,  # TODO: improve impl efficiency (currently zero embeddings are used -> disable completely!)
                inside_outside=False,  # only consider object-centric scenes
            ),
            bg_use_appearance_embedding=False,
            use_average_appearance_embedding=False,
            background_model="grid",
            eval_num_rays_per_chunk=1024,
            proposal_net_spatial_normalization_region="fg",
            feature_field=FeatureFieldConfig(
                hash_grid_max_res=128,
                use_position_encoding=True,  # smooth encoding
                use_fused_encoding_mlp=True
            ),
            feature_loss_mult=1.0,
            render_fg_seg=True,
            fg_seg_binary_thr=0.5,
            feature_seg_field=FeatureSegFieldConfig(
                knn=10,
                contrast_fg_bg=True
            )
        ),
    ),
    optimizers={
        "feature_field": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            "scheduler": None,
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)
