from . import *

# nesu-facto w/ explicit pointcloud regularization + separate plane nerf field modeling
neus_facto_wbg_reg_sep_plane_nerf = Config(
    method_name="neus-facto-wbg-reg_sep-plane-nerf",
    trainer=TrainerConfig(
        steps_per_eval_image=2500,
        steps_per_eval_batch=2500,
        steps_per_save=30000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=30001,
        mixed_precision=True,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=AutoReconDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=2048,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
            eval_camera_res_scale_factor=1.0,
            collate_fn=partial(
                nerfstudio_collate, extra_mappings={PCA: get_first_element, BasicPointClouds: get_first_element}
            ),
            train_mask_sample_start=10000,
            train_mask_sample_ratio=0.8,
        ),
        model=NeuSFactoRegModelConfig(
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
                hash_grid_progressive_training=False,
                hash_grid_progressive_training_iters=0,
                spatial_normalization_region="aabb",  # to better utilize the feature grid
                weight_norm=False,  # weight_norm might lead to NaNs during training
            ),
            # plane_sdf_field
            plane_field_type="nerf",
            plane_height_ratio=0.3,
            num_samples_plane=16,
            ptcd_reg_plane_field_mult=0.0,
            # explicit regularization
            ptcd_reg_fg_mult=0.1,
            ptcd_reg_plane_mult=0.1,
            ptcd_reg_n_iters=15000,
            ptcd_reg_weight_anneal_fn="cosine",
            mask_beta_prior_mult=0.0,
            handle_bottom_intersection_rays=True,
            cos_anneal_end=5000,
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
            # additional eikonal samples
            eikonal_loss_mult=0.05,
            num_proposal_samples_per_ray_for_eikonal_loss=64,
        ),
    ),
    optimizers={
        "proposal_networks": {  # fixed gamma=0.3 and n_milestones=3
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "proposal_networks_bg": {  # fixed gamma=0.3 and n_milestones=3
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15, weight_decay=1e-6),
            "scheduler": MultiStepSchedulerConfig(max_steps=30001),
        },
        "field_plane": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)


# nesu-facto w/ explicit pointcloud regularization + separate plane sdf field modeling
neus_facto_wbg_reg_sep_plane = Config(
    method_name="neus-facto-wbg-reg_sep-plane",
    trainer=TrainerConfig(
        steps_per_eval_image=2500,
        steps_per_eval_batch=2500,
        steps_per_save=30000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=30001,
        mixed_precision=True,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=AutoReconDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=2048,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
            eval_camera_res_scale_factor=1.0,
            collate_fn=partial(
                nerfstudio_collate, extra_mappings={PCA: get_first_element, BasicPointClouds: get_first_element}
            ),
            train_mask_sample_start=10000,
            train_mask_sample_ratio=0.8,
        ),
        model=NeuSFactoRegModelConfig(
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
                hash_grid_progressive_training=False,
                hash_grid_progressive_training_iters=0,
                spatial_normalization_region="aabb",  # to better utilize the feature grid
                weight_norm=False,  # weight_norm might lead to NaNs during training
            ),
            # plane_sdf_field
            plane_height_ratio=0.4,
            num_samples_plane=16,
            ptcd_reg_plane_field_mult=0.0,
            # explicit regularization
            ptcd_reg_fg_mult=0.1,
            ptcd_reg_plane_mult=0.1,
            ptcd_reg_n_iters=15000,
            ptcd_reg_weight_anneal_fn="cosine",
            mask_beta_prior_mult=0.0,
            handle_bottom_intersection_rays=True,
            cos_anneal_end=0,
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
        ),
    ),
    optimizers={
        "proposal_networks": {  # fixed gamma=0.3 and n_milestones=3
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "proposal_networks_bg": {  # fixed gamma=0.3 and n_milestones=3
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15, weight_decay=1e-6),
            "scheduler": MultiStepSchedulerConfig(max_steps=30001),
        },
        "field_plane": {
            "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15, weight_decay=1e-6),
            "scheduler": MultiStepSchedulerConfig(max_steps=30001),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)


# nesu-facto w/ explicit pointcloud regularization
neus_facto_wbg_reg = Config(
    method_name="neus-facto-wbg-reg",
    trainer=TrainerConfig(
        steps_per_eval_image=2500,
        steps_per_eval_batch=2500,
        steps_per_save=30000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=30001,
        mixed_precision=True,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=AutoReconDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=2048,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
            eval_camera_res_scale_factor=1.0,
            collate_fn=partial(
                nerfstudio_collate, extra_mappings={PCA: get_first_element, BasicPointClouds: get_first_element}
            ),
            train_mask_sample_start=10000,
            train_mask_sample_ratio=0.8,
        ),
        model=NeuSFactoModelConfig(
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
                hash_grid_progressive_training_iters=10000,
                spatial_normalization_region="aabb",  # to better utilize the feature grid
                weight_norm=False,  # weight_norm might lead to NaNs during training
            ),
            # explicit regularization
            ptcd_reg_fg_mult=0.1,
            ptcd_reg_plane_mult=0.1,
            ptcd_reg_n_iters=15000,
            ptcd_reg_weight_anneal_fn="cosine",
            mask_beta_prior_mult=0.0,
            handle_bottom_intersection_rays=False,
            cos_anneal_end=5000,
            proposal_net_spatial_normalization_region="fg",
            eval_num_rays_per_chunk=1024,
            use_average_appearance_embedding=False,
            background_model="grid",
            bg_use_appearance_embedding=False,
            use_fg_aware_scene_contraction=True,
            fg_aware_scene_contraction_alpha=5.0,
            bg_sampler_type="proposal_network",
            num_samples_outside=16,
            num_bg_proposal_samples_per_ray=(96, 64),
        ),
    ),
    optimizers={
        "proposal_networks": {  # fixed gamma=0.3 and n_milestones=3
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "proposal_networks_bg": {  # fixed gamma=0.3 and n_milestones=3
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15, weight_decay=1e-6),
            "scheduler": MultiStepSchedulerConfig(max_steps=30001),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)
