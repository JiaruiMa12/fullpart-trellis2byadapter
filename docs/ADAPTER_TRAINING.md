# Adapter Training Guide

This guide explains how to train the FullPart to TRELLIS2 SLat adapter, which converts FullPart sparse latents to TRELLIS2 format for high-quality 3D generation.

## Overview

The adapter is a lightweight neural network that maps:
- **Input**: FullPart sparse latents (8-channel, 64³ resolution)
- **Output**: TRELLIS2 sparse latents (32-channel, 32³ resolution)

The adapter uses cross-attention and coordinate embeddings to transform the latent space while preserving geometric structure.

## Prerequisites

### Data Preparation

1. **FullPart Latents**: Old sparse latents from FullPart encoder
   - Global latents: `dataset/partversexl/textured_mesh_latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/`
   - Part latents: `dataset/partversexl/textured_part_latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/`

2. **TRELLIS2 Latents**: Target latents from TRELLIS2 encoder
   - Global latents: `dataset/partversexl/trellis2_global_shape_latents/`
   - Part latents: `dataset/partversexl/trellis2_shape_part_latents/`

3. **Statistics File**: Normalization statistics
   - `dataset/partversexl/trellis2_shape_latents_stats.json`

4. **Dataset CSV**: List of asset IDs
   - Training: `dataset/partversexl/train.csv` or `dataset/partversexl/home_sample_1000.csv`
   - Validation: `dataset/partversexl/val.csv` or `dataset/partversexl/home_sample_1000.csv`

### Environment Setup

```bash
conda activate trellis2
export ATTN_BACKEND=xformers  # or sdpa, naive
```

## Training Configuration

### Global Adapter Training

Configuration file: `src/configs/train_configs/personal_configs_global_trellis2_adapter.py`

**Key Parameters**:
- `old_channels=8`: FullPart latent channels
- `trellis2_channels=32`: TRELLIS2 latent channels
- `hidden_dim=512`: Hidden dimension
- `num_heads=8`: Number of attention heads
- `num_layers=4`: Number of transformer layers
- `old_resolution=64`: FullPart resolution
- `trellis2_resolution=32`: TRELLIS2 resolution

**Training Settings**:
- Batch size: 1
- Learning rate: 5e-5
- Max grad norm: 0.1
- Precision: bf16
- Steps per save: 1000
- Steps per validation: 4000
- EMA decay: 0.9999

### Part Adapter Training

Configuration file: `src/configs/train_configs/personal_configs_part_trellis2.py`

**Key Parameters**:
- Similar to global adapter but with part-specific data loading
- Supports global-part joint training
- Rotation augmentation enabled

## Training Process

### 1. Start Training

```bash
cd /root/data/fullpart-main
conda activate trellis2
export ATTN_BACKEND=xformers

# Global adapter training
python main.py global_trellis2_adapter

# Part adapter training
python main.py 3dmaster_part_trellis2_adapter
```

### 2. Monitor Training

Training outputs are saved to:
- Checkpoints: `exps/[experiment_name]/checkpoints/`
- Logs: `exps/[experiment_name]/logs/`
- Samples: `sample_results/sample_results_global_trellis2/` or `sample_results/sample_results_trellis2/`

### 3. Resume from Checkpoint

Modify the config file to add checkpoint path or use CLI override:
```bash
python main.py global_trellis2_adapter --pipeline.ckpt_path=/path/to/checkpoint.ckpt
```

## Training Details

### Loss Function

The adapter is trained with MSE loss between predicted and target TRELLIS2 latents:
```
L = ||pred_feats - target_feats||²
```

### Data Normalization

- FullPart latents are normalized using pre-computed statistics
- TRELLIS2 latents are normalized using `trellis2_shape_latents_stats.json`
- Normalization is applied during training and must be reversed during inference

### Coordinate Transformation

The adapter remaps coordinates from 64³ to 32³ resolution:
```python
scale = trellis2_resolution / old_resolution  # 32 / 64 = 0.5
target_coords = floor(old_coords * scale)
```

### Architecture

1. **Input Projection**: Concatenates old features with Fourier coordinate embeddings
2. **Target Projection**: Projects target coordinates to query tokens
3. **Cross-Attention Layers**: 4 layers of cross-attention with feed-forward networks
4. **Output Projection**: Projects to TRELLIS2 feature dimension

## Troubleshooting

### Out of Memory
- Reduce batch size in config
- Reduce `dataloader_num_processes`
- Use gradient checkpointing

### NaN Loss
- Check data normalization
- Reduce learning rate
- Verify gradient clipping

### Poor Convergence
- Increase training steps
- Adjust learning rate
- Check data quality
- Verify TRELLIS2 latent paths

## Advanced Options

### Custom Configuration

Edit the config file to modify:
- Model architecture (hidden_dim, num_layers, etc.)
- Training hyperparameters (lr, batch_size, etc.)
- Data paths and dataset splits
- Validation frequency

### Multi-GPU Training

The trainer supports multi-GPU via DeepSpeed. Modify config:
```python
# In TrainerConfig
use_deepspeed=True
deepspeed_config="path/to/deepspeed_config.json"
```

### Mixed Precision

Default is bf16. Can change to fp16:
```python
training_precision="fp16"
```

## Evaluation

After training, evaluate on validation set:
```bash
python main.py global_trellis2_adapter --val_only
```

Metrics are logged to the experiment directory.

## Next Steps

After training the adapter:
1. Use the trained checkpoint for inference
2. See `docs/INFERENCE.md` for inference guide
3. Run `scripts/inference_adapter.py` to generate 3D models
