from . import *


# assume the fg object is bounded by a unit cube, and the entire scene is
# contracted to [-2, 2] as done in MipNeRF-360
neusfacto_autorecon = Config(
    method_name="neus-facto-autorecon",
    trainer=TrainerConfig(
        steps_per_eval_image=1000,
        steps_per_eval_batch=1000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=60001,  # 20k iters is not enough for high quality reconstruction
        mixed_precision=True,
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
                                               BasicPointClouds: get_first_element})
        ),
        model=NeuSFactoModelConfig(
            sdf_field=SDFFieldConfig(
                use_position_encoding=True,
                use_grid_feature=True,
                num_layers=2,
                num_layers_color=2,
                hidden_dim=256,
                bias=0.5,
                beta_init=0.3,
                direction_encoding_type="sh",
                use_appearance_embedding=False,
                inside_outside=False,
                spatial_normalization_region='full',  # [-2, 2] -> [0, 1] for feature grid interpolation
            ),
            cos_anneal_end=10000,
            background_model="none",
            eval_num_rays_per_chunk=1024,
            proposal_use_uniform_sampler=False,
            proposal_net_spatial_normalization_region="full"  # [-2, 2] -> [0, 1] for feature grid interpolation
        ),
    ),
    optimizers={
        "proposal_networks": {  # fixed gamma=0.3 and n_milestones=3
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=60001),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=1500, learning_rate_alpha=0.05, max_steps=60001),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)


# neus-facto with a BG Model
# This model has inferior fg reconstructions comparing to the neus-facto model.
neus_facto_wbg = Config(
    method_name="neus-facto-wbg",
    trainer=TrainerConfig(
        steps_per_eval_image=2500,
        steps_per_eval_batch=2500,
        steps_per_save=60000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=60001,
        mixed_precision=True,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            # TODO: set dataparser args
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
        model=NeuSFactoModelConfig(
            sdf_field=SDFFieldConfig(
                use_position_encoding=True,
                use_grid_feature=True,
                num_layers=2,
                num_layers_color=2,
                hidden_dim=256,
                bias=0.5,
                beta_init=0.3,
                direction_encoding_type="sh",
                use_appearance_embedding=False,  # TODO: improve impl efficiency (currently zero embeddings are used -> disable completely!)
                inside_outside=False,  # only consider object-centric scenes
            ),
            cos_anneal_end=10000,
            proposal_net_spatial_normalization_region="fg",
            eval_num_rays_per_chunk=1024,
            use_average_appearance_embedding=False,
            background_model="grid",
            bg_use_appearance_embedding=False,
            use_fg_aware_scene_contraction=True,
            fg_aware_scene_contraction_alpha=15.0,  # TODO: 15.0 -> 5.0
            bg_sampler_type="uniform_lin_disp",
            num_samples_outside=64,
        ),
    ),
    optimizers={
        "proposal_networks": {  # fixed gamma=0.3 and n_milestones=3
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=60001),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=1500, learning_rate_alpha=0.05, max_steps=60001),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=1500, learning_rate_alpha=0.05, max_steps=60001),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)


# neus-facto with a BG Model (w/ proposal-net sampler)
# This model has inferior fg reconstructions comparing to the neus-facto model.
neus_facto_wbg_BgProbNet = Config(
    method_name="neus-facto-wbg_bg-prop-net",
    trainer=TrainerConfig(
        steps_per_eval_image=2500,
        steps_per_eval_batch=2500,
        steps_per_save=60000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=60001,
        mixed_precision=True,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            # TODO: set dataparser args
            dataparser=AutoReconDataParserConfig(),  # NOTE: cannot set init args here, which would not be recognised.
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=2048,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
            eval_camera_res_scale_factor=1.0,
            collate_fn=partial(nerfstudio_collate,
                               extra_mappings={PCA: get_first_element,
                                               BasicPointClouds: get_first_element})
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
                use_appearance_embedding=False,  # TODO: improve impl efficiency (currently zero embeddings are used -> disable completely!)
                inside_outside=False,  # only consider object-centric scenes
                hash_grid_progressive_training=False,
                hash_grid_progressive_training_iters=0,  # TODO: run ablation
                spatial_normalization_region="aabb",  # to better utilize the feature grid
                weight_norm=False,  # weight_norm might lead to NaNs during training
            ),
            cos_anneal_end=0,  # FIXME: 5000
            proposal_net_spatial_normalization_region="fg",
            eval_num_rays_per_chunk=1024,
            use_average_appearance_embedding=False,
            background_model="grid",
            bg_use_appearance_embedding=False,
            use_fg_aware_scene_contraction=True,
            fg_aware_scene_contraction_alpha=5.0,
            bg_sampler_type="proposal_network",
            num_samples_outside=16,  # TODO: run ablation (use a smaller number to avoid the fg modeling affected by bg)
            num_bg_proposal_samples_per_ray=(96, 64)
        ),
    ),
    optimizers={
        "proposal_networks": {  # fixed gamma=0.3 and n_milestones=3
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            # "scheduler": MultiStepSchedulerConfig(max_steps=60001),
            "scheduler": None,
        },
        "proposal_networks_bg": {  # fixed gamma=0.3 and n_milestones=3
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
            # "scheduler": MultiStepSchedulerConfig(max_steps=30001),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            # "scheduler": NeuSSchedulerConfig(warm_up_end=1500, learning_rate_alpha=0.05, max_steps=60001),
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


# medium-sized model (deprecated)
neus_facto_wbg_medium = Config(
    method_name="neus-facto-wbg_medium",
    trainer=TrainerConfig(
        steps_per_eval_image=2000,
        steps_per_eval_batch=2000,
        steps_per_save=40000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=40001,
        mixed_precision=True,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            # TODO: set dataparser args
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
        model=NeuSFactoModelConfig(
            sdf_field=SDFFieldConfig(
                use_position_encoding=False,
                use_grid_feature=True,
                hash_grid_num_levels=16,
                num_layers=2,
                num_layers_color=2,
                hidden_dim=64,  # 256 -> 64
                geo_feat_dim=64,  # 256 -> 64
                direction_encoding_type="sh",
                hidden_dim_color=64,  # 256 -> 64
                color_network_include_sdf=False,
                bias=0.5,
                beta_init=0.3,
                use_appearance_embedding=False,  # TODO: improve impl efficiency (currently zero embeddings are used -> disable completely!)
                inside_outside=False,  # only consider object-centric scenes
            ),
            bg_use_appearance_embedding=False,
            use_average_appearance_embedding=False,
            background_model="grid",
            bg_hash_grid_num_levels=16,
            eval_num_rays_per_chunk=1024,
            proposal_net_spatial_normalization_region="fg"
        ),
    ),
    optimizers={
        "proposal_networks": {  # fixed gamma=0.3 and n_milestones=3
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=40001),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=1000, learning_rate_alpha=0.05, max_steps=40001),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=1000, learning_rate_alpha=0.05, max_steps=40001),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)


# tiny model (deprecated) - same model size as the one in NeuS2
neus_facto_wbg_tiny = Config(
    method_name="neus-facto-wbg_tiny",
    trainer=TrainerConfig(
        steps_per_eval_image=2000,
        steps_per_eval_batch=2000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=20001,
        mixed_precision=True,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            # TODO: set dataparser args
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
        model=NeuSFactoModelConfig(
            sdf_field=SDFFieldConfig(
                use_position_encoding=False,
                use_grid_feature=True,
                hash_grid_num_levels=14,  # NOTE: following NeuS2
                num_layers=1,  # NOTE: following NeuS2
                num_layers_color=2,
                hidden_dim=64,  # NOTE: following NeuS2
                geo_feat_dim=15,  # NOTE: following NeuS2
                direction_encoding_type="sh",  # NOTE: following NeuS2
                hidden_dim_color=64,  # NOTE: following NeuS2
                color_network_include_sdf=False,  # TODO: NeuS2 uses True
                bias=0.5,
                beta_init=0.3,
                use_appearance_embedding=False,  # TODO: improve impl efficiency (currently zero embeddings are used -> disable completely!)
                inside_outside=False,  # only consider object-centric scenes
            ),
            bg_use_appearance_embedding=False,
            use_average_appearance_embedding=False,
            background_model="grid",
            bg_hash_grid_num_levels=14,  # NOTE: following NeuS2
            eval_num_rays_per_chunk=1024,
            proposal_net_spatial_normalization_region="fg"
        ),
    ),
    optimizers={
        "proposal_networks": {  # fixed gamma=0.3 and n_milestones=3
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=20001),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)


# tiny model (deprecated) - same model size as the one in NeuS2 with a longer training schedule
neus_facto_wbg_tiny_long_schedule = Config(
    method_name="neus-facto-wbg_tiny_long-schedule",
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
        model=NeuSFactoModelConfig(
            sdf_field=SDFFieldConfig(
                use_position_encoding=True,
                use_grid_feature=True,
                hash_grid_num_levels=14,  # NOTE: following NeuS2
                num_layers=1,  # NOTE: following NeuS2
                num_layers_color=2,
                hidden_dim=64,  # NOTE: following NeuS2
                geo_feat_dim=15,  # NOTE: following NeuS2
                direction_encoding_type="sh",  # NOTE: following NeuS2
                hidden_dim_color=64,  # NOTE: following NeuS2
                color_network_include_sdf=False,  # TODO: NeuS2 uses True
                bias=0.5,
                beta_init=0.3,
                use_appearance_embedding=False,  # TODO: improve impl efficiency (currently zero embeddings are used -> disable completely!)
                inside_outside=False,  # only consider object-centric scenes
            ),
            cos_anneal_end=5000,
            use_fg_aware_scene_contraction=True,
            fg_aware_scene_contraction_alpha=15.0,
            bg_sampler_type="uniform_lin_disp",  # TODO: ablation: "proposal_network"
            num_samples_outside=64,
            bg_use_appearance_embedding=False,
            use_average_appearance_embedding=False,
            background_model="grid",
            bg_hash_grid_num_levels=14,  # NOTE: following NeuS2
            eval_num_rays_per_chunk=1024,
            proposal_net_spatial_normalization_region="fg"
        ),
    ),
    optimizers={
        "proposal_networks": {  # fixed gamma=0.3 and n_milestones=3
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=30001),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=750, learning_rate_alpha=0.05, max_steps=30001),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=750, learning_rate_alpha=0.05, max_steps=30001),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)
