#!/bin/bash
# Inference script for FullPart to TRELLIS2 Adapter Pipeline
#
# This script runs the full inference pipeline: FullPart Stage1 -> Stage2 -> Adapter -> TRELLIS2
#
# Usage:
#   ./inference_adapter.sh <image_path> [options]
#
# Required arguments:
#   <image_path>        Path to input image (e.g., assets/demo_examples/robot.png)
#
# Optional arguments:
#   --box-path          Path to bounding box file (default: auto-detect from image directory)
#   --s1-ckpt           Stage1 checkpoint path (default: pretrained_models/fullpart/ckpts/s1.ckpt)
#   --s2-ckpt           Stage2 checkpoint path (default: pretrained_models/fullpart/ckpts/s2.ckpt)
#   --adapter-ckpt      Adapter checkpoint path (required)
#   --output-dir        Output directory (default: outputs/demo_inference)
#   --skip-stage1       Skip Stage1 (use existing outputs)
#   --skip-stage2       Skip Stage2 (use existing outputs)
#   --skip-adapter      Skip Adapter + TRELLIS2
#   --skip-assembly     Skip assembly
#   --texture-size      Texture size (default: 512)
#   --decimation-target Mesh decimation target (default: 100000)
#   --max-old-tokens    Max old tokens (default: 1500)
#   --part-ids          Specific part IDs to decode (space-separated)
#   --device            Device (default: cuda)
#
# Examples:
#   # Basic inference
#   ./inference_adapter.sh assets/demo_examples/robot.png \
#       --adapter-ckpt exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt
#
#   # Skip Stage1 (use existing outputs)
#   ./inference_adapter.sh assets/demo_examples/robot.png \
#       --adapter-ckpt exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
#       --skip-stage1
#
#   # Custom output directory
#   ./inference_adapter.sh assets/demo_examples/robot.png \
#       --adapter-ckpt exps/unnamed/checkpoints/checkpoint-75000/slat_adapter/pytorch_model.ckpt \
#       --output-dir outputs/my_inference

set -e  # Exit on error

# Default values
IMAGE_PATH=""
BOX_PATH=""
S1_CKPT="pretrained_models/fullpart/ckpts/s1.ckpt"
S2_CKPT="pretrained_models/fullpart/ckpts/s2.ckpt"
ADAPTER_CKPT=""
OUTPUT_DIR="outputs/demo_inference"
SKIP_STAGE1=false
SKIP_STAGE2=false
SKIP_ADAPTER=false
SKIP_ASSEMBLY=false
TEXTURE_SIZE=512
DECIMATION_TARGET=100000
MAX_OLD_TOKENS=1500
PART_IDS=""
DEVICE="cuda"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --box-path)
            BOX_PATH="$2"
            shift 2
            ;;
        --s1-ckpt)
            S1_CKPT="$2"
            shift 2
            ;;
        --s2-ckpt)
            S2_CKPT="$2"
            shift 2
            ;;
        --adapter-ckpt)
            ADAPTER_CKPT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-stage1)
            SKIP_STAGE1=true
            shift
            ;;
        --skip-stage2)
            SKIP_STAGE2=true
            shift
            ;;
        --skip-adapter)
            SKIP_ADAPTER=true
            shift
            ;;
        --skip-assembly)
            SKIP_ASSEMBLY=true
            shift
            ;;
        --texture-size)
            TEXTURE_SIZE="$2"
            shift 2
            ;;
        --decimation-target)
            DECIMATION_TARGET="$2"
            shift 2
            ;;
        --max-old-tokens)
            MAX_OLD_TOKENS="$2"
            shift 2
            ;;
        --part-ids)
            shift
            PART_IDS=""
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                if [ -z "$PART_IDS" ]; then
                    PART_IDS="$1"
                else
                    PART_IDS="$PART_IDS $1"
                fi
                shift
            done
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            if [ -z "$IMAGE_PATH" ]; then
                IMAGE_PATH="$1"
            else
                echo "Error: Unknown argument '$1'"
                echo "Usage: $0 <image_path> [options]"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check required arguments
if [ -z "$IMAGE_PATH" ]; then
    echo "Error: Image path is required"
    echo "Usage: $0 <image_path> [options]"
    exit 1
fi

if [ -z "$ADAPTER_CKPT" ] && [ "$SKIP_ADAPTER" = false ]; then
    echo "Error: --adapter-ckpt is required (unless --skip-adapter is set)"
    echo "Usage: $0 <image_path> --adapter-ckpt <path> [options]"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment: trellis2"
conda activate trellis2

# Set spconv algorithm to native for stability (required for Stage2)
export SPCONV_ALGO=native

echo "=========================================="
echo "FullPart to TRELLIS2 Adapter Inference"
echo "=========================================="
echo "Image path: $IMAGE_PATH"
echo "Box path: ${BOX_PATH:-auto-detect}"
echo "Stage1 checkpoint: $S1_CKPT"
echo "Stage2 checkpoint: $S2_CKPT"
echo "Adapter checkpoint: ${ADAPTER_CKPT:-none (skipped)}"
echo "Output directory: $OUTPUT_DIR"
echo "Skip Stage1: $SKIP_STAGE1"
echo "Skip Stage2: $SKIP_STAGE2"
echo "Skip Adapter: $SKIP_ADAPTER"
echo "Skip Assembly: $SKIP_ASSEMBLY"
echo "Texture size: $TEXTURE_SIZE"
echo "Decimation target: $DECIMATION_TARGET"
echo "Max old tokens: $MAX_OLD_TOKENS"
echo "Part IDs: ${PART_IDS:-all}"
echo "Device: $DEVICE"
echo "=========================================="

# Change to project directory
cd /root/data/fullpart-main

# Build command
CMD="python scripts/inference_adapter.py"
CMD="$CMD --raw-path $IMAGE_PATH"
CMD="$CMD --stage1.transformer-ckpt $S1_CKPT"
CMD="$CMD --stage2.transformer-ckpt $S2_CKPT"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --texture-size $TEXTURE_SIZE"
CMD="$CMD --decimation-target $DECIMATION_TARGET"
CMD="$CMD --max-old-tokens $MAX_OLD_TOKENS"
CMD="$CMD --device $DEVICE"

if [ -n "$BOX_PATH" ]; then
    CMD="$CMD --raw-box $BOX_PATH"
fi

if [ -n "$ADAPTER_CKPT" ]; then
    CMD="$CMD --adapter-ckpt $ADAPTER_CKPT"
fi

if [ "$SKIP_STAGE1" = true ]; then
    CMD="$CMD --skip-stage1"
fi

if [ "$SKIP_STAGE2" = true ]; then
    CMD="$CMD --skip-stage2"
fi

if [ "$SKIP_ADAPTER" = true ]; then
    CMD="$CMD --skip-adapter"
fi

if [ "$SKIP_ASSEMBLY" = true ]; then
    CMD="$CMD --skip-assembly"
fi

if [ -n "$PART_IDS" ]; then
    CMD="$CMD --decode-part-ids $PART_IDS"
fi

echo "Running command:"
echo "$CMD"
echo "=========================================="

# Run inference
eval $CMD

echo "=========================================="
echo "Inference completed successfully!"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
