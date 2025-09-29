from . import *

# TODO: neus w/o hash grid + NeRF++ scene parameterization

# neus (MLP-based) w/ a BG Model (l2 scene contraction)
neus_wbg_mlp = Config(
    method_name="neus-wbg_mlp",
    trainer=TrainerConfig(
        steps_per_eval_image=2500,
        steps_per_eval_batch=2500,
        steps_per_save=50000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=300000,
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=512,
            eval_num_rays_per_batch=512,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuSModelConfig(
            sdf_field=SDFFieldConfig(
                use_position_encoding=True,
                bias=0.2,
                beta_init=0.3,
                inside_outside=False,
            ),
            scene_contraction_order="none",
            bg_sampler_type="lin_disp",  # TODO: ablate "uniform_lin_disp"
            background_model="mlp",
            use_fg_aware_scene_contraction=False,  # TODO: True might lead to better results
            bg_use_appearance_embedding=False,
            use_average_appearance_embedding=False,
            eval_num_rays_per_chunk=1024
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)


# (deprecated) neus with a BG Model
neus_wbg = Config(
    method_name="neus-wbg",
    trainer=TrainerConfig(
        steps_per_eval_image=1000,
        steps_per_eval_batch=1000,
        steps_per_save=20000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=60001,  # TODO: 20001 is enough?
        mixed_precision=False,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SDFStudioDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NeuSModelConfig(
            sdf_field=SDFFieldConfig(
                use_grid_feature=True,
                num_layers=2,
                num_layers_color=2,
                hidden_dim=256,
                bias=0.5,
                beta_init=0.3,
                use_appearance_embedding=False,  # TODO: improve impl efficiency (currently zero embeddings are used -> disable completely!)
                inside_outside=False,  # only consider object-centric scenes
            ),
            cos_anneal_end=10000,
            bg_use_appearance_embedding=False,
            use_average_appearance_embedding=False,
            background_model="grid",
            # TODO: check bg model size
            eval_num_rays_per_chunk=1024,
        ),
    ),
    optimizers={  # TODO: use higher lr for grid feature
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=1500, learning_rate_alpha=0.05, max_steps=60001),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": NeuSSchedulerConfig(warm_up_end=1500, learning_rate_alpha=0.05, max_steps=60001),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)
