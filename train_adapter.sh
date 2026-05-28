#!/bin/bash
# Training script for FullPart to TRELLIS2 SLat Adapter
#
# This script trains the adapter that converts FullPart sparse latents to TRELLIS2 format.
# Two training modes are available:
#   - Global adapter: Converts global mesh latents
#   - Part adapter: Converts part-level latents
#
# Usage:
#   ./train_adapter.sh [global|part]
#
# Examples:
#   ./train_adapter.sh global    # Train global adapter
#   ./train_adapter.sh part      # Train part adapter

set -e  # Exit on error

# Default training mode
TRAINING_MODE=${1:-global}

# Activate conda environment
echo "Activating conda environment: trellis2"
conda activate trellis2

# Set attention backend (options: xformers, sdpa, naive)
# xformers is recommended for best performance
export ATTN_BACKEND=xformers

# Set spconv algorithm to native for stability
export SPCONV_ALGO=native

echo "=========================================="
echo "Adapter Training Script"
echo "=========================================="
echo "Training mode: $TRAINING_MODE"
echo "Attention backend: $ATTN_BACKEND"
echo "Spconv algorithm: $SPCONV_ALGO"
echo "=========================================="

# Change to project directory
cd /root/data/fullpart-main

if [ "$TRAINING_MODE" = "global" ]; then
    echo "Starting global adapter training..."
    echo "Config: src/configs/train_configs/personal_configs_global_trellis2_adapter.py"
    echo "Experiment name: global_trellis2_adapter"
    
    python main.py global_trellis2_adapter
    
elif [ "$TRAINING_MODE" = "part" ]; then
    echo "Starting part adapter training..."
    echo "Config: src/configs/train_configs/personal_configs_part_trellis2.py"
    echo "Experiment name: 3dmaster_part_trellis2_adapter"
    
    python main.py 3dmaster_part_trellis2_adapter
    
else
    echo "Error: Invalid training mode '$TRAINING_MODE'"
    echo "Usage: $0 [global|part]"
    exit 1
fi

echo "=========================================="
echo "Training completed successfully!"
echo "=========================================="
echo "Checkpoints saved to: exps/[experiment_name]/checkpoints/"
echo "Logs saved to: exps/[experiment_name]/logs/"
echo "Samples saved to: sample_results/"
echo "=========================================="
