from . import *


# 2-stages training of neus-facto:
# 1. train nerfacto; (nerfacto is very robust to the scene condition)
# 2. replace fg nerfacto_field with fg sdf_field (optionally supervised by pre-trained nerfacto)
distilled_neus_facto_wbg = Config(
    method_name="distilled-neus-facto-wbg",
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
            train_num_rays_per_batch=4096,  # 2048,
            eval_num_rays_per_batch=4096,  # 1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
            eval_camera_res_scale_factor=1.0,
            collate_fn=partial(nerfstudio_collate,
                               extra_mappings={PCA: get_first_element,
                                               BasicPointClouds: get_first_element})
        ),
        model=DistilledNeuSFactoModelConfig(
            nerfacto_field_enabled=True,  # for the 1st-stage training
            nerfacto_spatial_normalization_region="fg",  # bg nerfacto always uses full
            proposal_net_use_separate_contraction=True,
            proposal_net_contraction_scale_factor=0.25,
            proposal_net_spatial_normalization_region="full",
            proposal_use_uniform_sampler=False,
            use_fg_aware_scene_contraction=True,
            fg_aware_scene_contraction_alpha=15.0,
            sdf_field=SDFFieldConfig(  # NeuSFacto Medium
                use_position_encoding=True,
                use_grid_feature=True,
                num_layers=2,
                num_layers_color=2,
                hidden_dim=64,  # 256 -> 64
                geo_feat_dim=64,  # 256 -> 64 -> 15
                bias=0.5,
                beta_init=0.3,
                direction_encoding_type="sh",
                use_appearance_embedding=False,  # TODO: improve impl efficiency (currently zero embeddings are used -> disable completely!)
                inside_outside=False,  # only consider object-centric scenes
            ),
            cos_anneal_end=0,  # TODO: set 5000 while training NeuS
            eval_num_rays_per_chunk=1024,
            use_average_appearance_embedding=False,
            background_model="grid",
            bg_use_appearance_embedding=False,
            bg_sampler_type="none",
            num_samples_outside=0,
        ),
    ),
    optimizers={
        "proposal_networks": {  # TODO: disable this optimizer during 2nd-stage training
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {  # for NerfactoField / SDFField
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "field_background": {  # TODO: disable this optimizer during 2nd-stage training
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

# try to replicate the result of nerfacto (using the nerfacto field in fg & bg)
debug_distilled_neus_facto_wbg = Config(
    method_name="debug_distilled-neus-facto-wbg",
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
            train_num_rays_per_batch=4096,  # 2048,
            eval_num_rays_per_batch=4096,  # 1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
            eval_camera_res_scale_factor=1.0,
            collate_fn=partial(nerfstudio_collate,
                               extra_mappings={PCA: get_first_element,
                                               BasicPointClouds: get_first_element})
        ),
        model=DistilledNeuSFactoModelConfig(
            nerfacto_field_enabled=True,  # for the 1st-stage training
            nerfacto_spatial_normalization_region="full",  # bg nerfacto always uses full
            single_field_fg_bg=True,
            proposal_net_use_separate_contraction=True,  # proposal net scene contraction
            proposal_net_contraction_scale_factor=1.0,
            proposal_net_spatial_normalization_region="full",
            proposal_use_uniform_sampler=False,
            use_fg_aware_scene_contraction=False,  # main field scene contraction
            scene_contraction_scale_factor=1.0,
            sdf_field=SDFFieldConfig(  # NeuSFacto Medium
                use_position_encoding=True,
                use_grid_feature=True,
                num_layers=2,
                num_layers_color=2,
                hidden_dim=64,  # 256 -> 64
                geo_feat_dim=64,  # 256 -> 64 -> 15
                bias=0.5,
                beta_init=0.3,
                direction_encoding_type="sh",
                use_appearance_embedding=False,  # TODO: improve impl efficiency (currently zero embeddings are used -> disable completely!)
                inside_outside=False,  # only consider object-centric scenes
            ),
            cos_anneal_end=0,  # TODO: set 5000 while training NeuS
            eval_num_rays_per_chunk=1024,
            use_average_appearance_embedding=False,
            background_model="grid",
            bg_use_appearance_embedding=False,
            bg_sampler_type="none",
            num_samples_outside=0,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {  # for NerfactoField
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "field_background": {  # NOTE: comment out this part if single_field_fg_bg is True
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)
