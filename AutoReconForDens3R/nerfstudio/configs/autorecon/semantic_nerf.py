from . import *


autorecon_semantic_nerf = Config(
    method_name="autorecon_semantic-nerfw",
    trainer=TrainerConfig(
        steps_per_eval_batch=500, steps_per_eval_image=500,
        steps_per_save=30000, max_num_iterations=30001, mixed_precision=True
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=AutoReconDataManagerConfig(
            dataparser=AutoReconDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096
        ),
        model=SemanticNerfWModelConfig(
            eval_num_rays_per_chunk=1 << 16,
            semantic_mult=1.0,
            use_transient_embedding=False,
            scene_contraction_scale_factor=0.25,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 16),
    vis="viewer",
)
