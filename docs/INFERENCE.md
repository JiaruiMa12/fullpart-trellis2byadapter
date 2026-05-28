# Inference Guide

This guide explains how to use the trained adapter to generate 3D models from images using the FullPart + TRELLIS2 pipeline.

## Overview

The inference pipeline consists of the following stages:
1. **FullPart Stage1**: Generate coarse part voxel grids from image
2. **FullPart Stage2**: Generate global and part sparse latents
3. **Adapter Conversion**: Convert FullPart latents to TRELLIS2 format
4. **TRELLIS2 Shape Refinement**: Refine shape using flow matching
5. **TRELLIS2 Texture Decoding**: Generate textures
6. **Assembly**: Assemble part GLBs into final 3D model

## Prerequisites

### Required Checkpoints

1. **FullPart Stage1**: Transformer checkpoint for voxel generation
   - Example: `pretrained_models/fullpart/ckpts/s1.ckpt`

2. **FullPart Stage2**: Transformer checkpoint for sparse latent generation
   - Example: `pretrained_models/fullpart/ckpts/s2.ckpt`

3. **Adapter**: Trained adapter checkpoint
   - Example: `exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt`

4. **TRELLIS2**: Pretrained TRELLIS2 model
   - Path: `/root/data/pretrained_models/trellis.2-4B/ckpts`

### Input Data

1. **Image**: PNG/JPG image of the object
   - Example: `assets/demo_examples/robot.png`

2. **Bounding Box**: NPY/JSON file containing part bounding boxes
   - Example: `assets/demo_examples/robot.npy`
   - Format: `(num_parts, 2, 3)` array with min/max coordinates

### Environment Setup

```bash
conda activate trellis2
export SPCONV_ALGO=native  # Required for Stage2
```

## Inference Methods

### Method 1: Full Pipeline (inference_adapter.py)

This is the recommended method for end-to-end inference.

**Script**: `scripts/inference_adapter.py`

**Basic Usage**:
```bash
python scripts/inference_adapter.py \
    --raw-path assets/demo_examples/robot.png \
    --raw-box assets/demo_examples/robot.npy \
    --stage1.transformer-ckpt pretrained_models/fullpart/ckpts/s1.ckpt \
    --stage2.transformer-ckpt pretrained_models/fullpart/ckpts/s2.ckpt \
    --adapter-ckpt exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
    --output-dir outputs/demo_inference
```

**Key Parameters**:
- `--raw-path`: Path to input image
- `--raw-box`: Path to bounding box file (optional, auto-detected if same directory)
- `--stage1.transformer-ckpt`: Stage1 checkpoint path
- `--stage2.transformer-ckpt`: Stage2 checkpoint path
- `--adapter-ckpt`: Adapter checkpoint path (required)
- `--output-dir`: Output directory

**Skip Stages**:
```bash
# Skip Stage1 (use existing Stage1 outputs)
python scripts/inference_adapter.py \
    --raw-path assets/demo_examples/robot.png \
    --skip-stage1 \
    --stage1.transformer-ckpt pretrained_models/fullpart/ckpts/s1.ckpt \
    --stage2.transformer-ckpt pretrained_models/fullpart/ckpts/s2.ckpt \
    --adapter-ckpt exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
    --output-dir outputs/demo_inference

# Skip Stage2 (use existing Stage2 outputs)
python scripts/inference_adapter.py \
    --raw-path assets/demo_examples/robot.png \
    --skip-stage2 \
    --stage1.transformer-ckpt pretrained_models/fullpart/ckpts/s1.ckpt \
    --stage2.transformer-ckpt pretrained_models/fullpart/ckpts/s2.ckpt \
    --adapter-ckpt exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
    --output-dir outputs/demo_inference

# Skip Adapter + TRELLIS2 (only run FullPart)
python scripts/inference_adapter.py \
    --raw-path assets/demo_examples/robot.png \
    --skip-adapter \
    --skip-trellis2 \
    --stage1.transformer-ckpt pretrained_models/fullpart/ckpts/s1.ckpt \
    --stage2.transformer-ckpt pretrained_models/fullpart/ckpts/s2.ckpt \
    --output-dir outputs/demo_inference
```

**TRELLIS2 Parameters**:
```bash
python scripts/inference_adapter.py \
    --raw-path assets/demo_examples/robot.png \
    --stage1.transformer-ckpt pretrained_models/fullpart/ckpts/s1.ckpt \
    --stage2.transformer-ckpt pretrained_models/fullpart/ckpts/s2.ckpt \
    --adapter-ckpt exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
    --texture-size 512 \
    --decimation-target 100000 \
    --max-old-tokens 1500 \
    --output-dir outputs/demo_inference
```

**Selective Part Decoding**:
```bash
# Only decode specific parts (e.g., parts 0, 2, 5)
python scripts/inference_adapter.py \
    --raw-path assets/demo_examples/robot.png \
    --stage1.transformer-ckpt pretrained_models/fullpart/ckpts/s1.ckpt \
    --stage2.transformer-ckpt pretrained_models/fullpart/ckpts/s2.ckpt \
    --adapter-ckpt exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
    --decode-part-ids 0 2 5 \
    --output-dir outputs/demo_inference
```

### Method 2: Part-Level Inference (part_bbox_adapter_trellis2.py)

Use this method when you already have Stage2 outputs and want to process specific parts.

**Script**: `scripts/part_bbox_adapter_trellis2.py`

**Basic Usage**:
```bash
python scripts/part_bbox_adapter_trellis2.py \
    --asset-id 001e4438fc2742ad81356aae14c94d4a \
    --anno-dir dataset/partversexl/anno_infos \
    --renders-cond-dir dataset/partversexl/renders_cond \
    --adapter-ckpt exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
    --output-dir outputs/part_inference
```

**Using Old Part Latents**:
```bash
python scripts/part_bbox_adapter_trellis2.py \
    --asset-id 001e4438fc2742ad81356aae14c94d4a \
    --old-part-dir /mjr/textured_part_latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16 \
    --adapter-ckpt exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
    --output-dir outputs/part_inference
```

**Using Stage2 Part SLats**:
```bash
python scripts/part_bbox_adapter_trellis2.py \
    --asset-id 001e4438fc2742ad81356aae14c94d4a \
    --use-part-slats \
    --stage2-output-dir outputs/demo_inference/stage2 \
    --adapter-ckpt exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
    --output-dir outputs/part_inference
```

**Shape Flow Refinement**:
```bash
# Enable shape flow refinement with custom parameters
python scripts/part_bbox_adapter_trellis2.py \
    --asset-id 001e4438fc2742ad81356aae14c94d4a \
    --adapter-ckpt exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
    --refine-with-shape-flow \
    --shape-flow-steps 25 \
    --shape-flow-start-t 0.9 \
    --shape-init-noise-strength 0.0 \
    --output-dir outputs/part_inference
```

**Shape Flow Parameters**:
- `--refine-with-shape-flow`: Enable TRELLIS2 shape flow refinement
- `--shape-flow-steps`: Number of flow steps (default: 12)
- `--shape-flow-start-t`: Starting timestep t ∈ [0, 1] (default: 1.0)
  - 1.0: Full denoise from pure noise schedule
  - 0.9: Start from 90% noise (light refinement)
  - 0.5: Start from 50% noise (moderate refinement)
- `--shape-init-noise-strength`: Noise strength for adapter output (default: 0.0)

**Selective Part Processing**:
```bash
# Only process specific parts
python scripts/part_bbox_adapter_trellis2.py \
    --asset-id 001e4438fc2742ad81356aae14c94d4a \
    --adapter-ckpt exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
    --part-ids 0 2 5 7 \
    --output-dir outputs/part_inference
```

**Other Parameters**:
```bash
python scripts/part_bbox_adapter_trellis2.py \
    --asset-id 001e4438fc2742ad81356aae14c94d4a \
    --adapter-ckpt exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
    --resolution 512 \
    --bbox-padding 0.02 \
    --min-tokens 64 \
    --decimation-target 200000 \
    --texture-size 1024 \
    --output-dir outputs/part_inference
```

### Method 3: Batch Inference

Process multiple assets from a CSV file:

```bash
python scripts/part_bbox_adapter_trellis2.py \
    --csv dataset/partversexl/home_sample_1000.csv \
    --adapter-ckpt exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
    --output-dir outputs/batch_inference
```

## Output Structure

### Full Pipeline Output

```
outputs/demo_inference/
├── stage1/
│   └── {asset_id}/
│       ├── part_00_voxel.npz
│       ├── part_01_voxel.npz
│       └── ...
├── stage2/
│   └── {asset_id}/
│       ├── global_slat.npz
│       ├── part_00_slat.npz
│       ├── part_01_slat.npz
│       └── ...
└── trellis2/
    └── {asset_id}/
        ├── part_00_adapter_trellis2.glb
        ├── part_01_adapter_trellis2.glb
        ├── part_00_adapter_shape_slat_norm.npz
        ├── part_01_adapter_shape_slat_norm.npz
        └── summary.json
```

### Part-Level Output

```
outputs/part_inference/
└── {asset_id}/
    ├── part_00_adapter_trellis2.glb
    ├── part_01_adapter_trellis2.glb
    ├── part_00_adapter_shape_slat_norm.npz
    ├── part_01_adapter_shape_slat_norm.npz
    └── summary.json
```

## Troubleshooting

### Out of Memory

**Symptoms**: CUDA OOM error during inference

**Solutions**:
- Reduce `--texture-size` (e.g., 512 → 256)
- Reduce `--decimation-target` (e.g., 200000 → 100000)
- Reduce `--max-old-tokens` (e.g., 1500 → 1000)
- Process fewer parts at a time with `--part-ids`
- Use lower resolution: `--resolution 512`

### SIGFPE Error

**Symptoms**: Floating point exception during Stage2

**Solution**:
```bash
export SPCONV_ALGO=native
```

### Poor Quality Output

**Symptoms**: Generated 3D models have artifacts or low quality

**Solutions**:
- Increase `--shape-flow-steps` (e.g., 12 → 25)
- Adjust `--shape-flow-start-t` (e.g., 1.0 → 0.9 for lighter refinement)
- Check adapter checkpoint quality
- Verify input image and bounding box quality
- Try different `--shape-init-noise-strength` values

### Missing Bounding Box

**Symptoms**: Error loading bounding box file

**Solution**:
- Ensure `.npy` or `.json` file exists in same directory as image
- Check file format: `(num_parts, 2, 3)` array
- Provide explicit path with `--raw-box`

### Adapter Checkpoint Not Found

**Symptoms**: Error loading adapter checkpoint

**Solution**:
- Verify checkpoint path is correct
- Check that checkpoint contains `slat_adapter` state dict
- Use `--adapter-ckpt` parameter explicitly

## Advanced Usage

### Custom TRELLIS2 Configuration

```bash
python scripts/inference_adapter.py \
    --raw-path assets/demo_examples/robot.png \
    --stage1.transformer-ckpt pretrained_models/fullpart/ckpts/s1.ckpt \
    --stage2.transformer-ckpt pretrained_models/fullpart/ckpts/s2.ckpt \
    --adapter-ckpt exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
    --trellis2-root /root/data/TRELLIS.2 \
    --trellis2-pretrained /root/data/pretrained_models/trellis.2-4B/ckpts \
    --config-file pipeline_local.json \
    --output-dir outputs/demo_inference
```

### Assembly

Assemble part GLBs into a single model:

```bash
python scripts/assemble_part_glbs.py \
    --asset-id 001e4438fc2742ad81356aae14c94d4a \
    --part-dir outputs/demo_inference/trellis2/001e4438fc2742ad81356aae14c94d4a \
    --source-part-dir /mjr/textured_part_glbs \
    --output outputs/demo_inference/001e4438fc2742ad81356aae14c94d4a_assembled.glb
```

## Performance Tips

1. **Use Stage2 SLats**: Skip Stage2 mesh decoding with `--skip-stage2-decode` to save time
2. **Selective Parts**: Process only needed parts with `--part-ids` or `--decode-part-ids`
3. **Lower Resolution**: Use `--resolution 512` instead of 1024 for faster inference
4. **Batch Processing**: Use CSV file for batch inference
5. **Shape Flow Start T**: Use `--shape-flow-start-t 0.9` for faster refinement with minimal quality loss

## Next Steps

- See `docs/ADAPTER_TRAINING.md` for training the adapter
- Check `scripts/` for additional utility scripts
- Review experiment outputs in `exps/` for training logs
