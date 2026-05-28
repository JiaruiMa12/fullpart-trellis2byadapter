from typing import Dict, List
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "src/submodule/TRELLIS"))

from ...data import GlobalSLatPairDataConfig
from ...engine import AdamWOptimizerConfig, SchedulerConfig, StabilityConfig, TrainerConfig
from ...models import *
from ...pipelines import *


train_dataset_csv_path = "dataset/partversexl/home_sample_1000.csv"
eval_dataset_csv_path = "dataset/partversexl/home_sample_1000.csv"
old_global_slat_dir = "dataset/partversexl/textured_mesh_latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
trellis2_global_slat_dir = "dataset/partversexl/trellis2_global_shape_latents"
trellis2_shape_stats_json = "dataset/partversexl/trellis2_shape_latents_stats.json"


slat_adapter_config = FullPartToTrellis2SLatAdapterConfig(
    old_channels=8,
    trellis2_channels=32,
    hidden_dim=512,
    num_heads=8,
    num_layers=4,
    old_resolution=64,
    trellis2_resolution=32,
)


def config(
    use_ema: bool = True,
    batch_size: int = 1,
    lr: float = 5e-5,
    max_grad_norm: float = 0.1,
    training_precision: str = "bf16",
    trainable_modules: List = ["slat_adapter"],
    nontrainable_modules: List = [],
    dataloader_num_processes: int = 2,
    step_per_save: int = 1000,
    slat_norm: bool = True,
):
    return TrainerConfig(
        num_epochs=10000,
        training_precision=training_precision,
        step_per_save=step_per_save,
        step_per_val=4000,
        val_at_begin=False,
        val_types=[],
        use_ema=use_ema,
        save_ema=False,
        step_per_ema=1,
        ema_decay=0.9999,
        ema_start_step=0,
        seed=None,
        delete_deepspeed_weights=True,
        train_data=GlobalSLatPairDataConfig(
            dataset_csv_path=train_dataset_csv_path,
            old_global_slat_dir=old_global_slat_dir,
            trellis2_global_slat_dir=trellis2_global_slat_dir,
            trellis2_slat_stats_json=trellis2_shape_stats_json,
            batch_size=batch_size,
            num_processes=dataloader_num_processes,
            num_samples=200,#200
            shuffle=True,
            slat_norm=slat_norm,
            trellis2_slat_norm=True,
        ),
        joint_train_prob=0,
        val_data=GlobalSLatPairDataConfig(
            dataset_csv_path=eval_dataset_csv_path,
            old_global_slat_dir=old_global_slat_dir,
            trellis2_global_slat_dir=trellis2_global_slat_dir,
            trellis2_slat_stats_json=trellis2_shape_stats_json,
            batch_size=1,
            num_processes=1,
            shuffle=False,
            num_samples=8,#8
            slat_norm=slat_norm,
            trellis2_slat_norm=True,
        ),
        trainable_modules=trainable_modules,
        nontrainable_modules=nontrainable_modules,
        optimizer=AdamWOptimizerConfig(fused=True, lr=lr, max_grad_norm=max_grad_norm),
        scheduler=SchedulerConfig(),
        stability=StabilityConfig(stability_protection=True),
        pipeline=JointDiTSingle3DPipelineConfigStage2Trellis2(
            ckpt_path=None,
            vae_config=None,
            voxel_vae_config=None,
            slat_adapter_config=slat_adapter_config,
            transformer_config=None,
            call={"save_dir": "./sample_results/sample_results_global_trellis2"},
            trellis2_root="/root/data/TRELLIS.2",
            trellis2_pretrained="/root/data/pretrained_models/trellis.2-4B/ckpts",
            trellis2_pipeline_config_file="pipeline_local.json",
            trellis2_shape_resolution=512,
            trellis2_flow_init_strength=0.5,
            use_trellis2_flow_refine=False,
        ),
    )


personal_configs_global_trellis2_adapter: Dict[str, TrainerConfig] = {}

if JointDiTSingle3DPipelineConfigStage2Trellis2 is not None:
    personal_configs_global_trellis2_adapter["global_trellis2_adapter"] = config()
