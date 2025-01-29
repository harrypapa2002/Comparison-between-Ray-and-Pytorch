#!/bin/bash

# ============================================================
# Launch Script for PyTorch Distributed Lesion Classification
# Master Node: namenode
# Processes: 4 (one per CPU core)
# ============================================================

# ----------------------------
# Configuration Parameters
# ----------------------------

# Master Node Details
MASTER_ADDR="namenode"
MASTER_PORT=29500
NNODES=1
NPROC_PER_NODE=4
WORLD_SIZE=$((NNODES * NPROC_PER_NODE)) 

# PyTorch Script Path (absolute path)
PYTORCH_SCRIPT_PATH="/home/ubuntu/Comparison-between-Ray-and-Pytorch/LesionClassification/lesion_classification_pytorch.py"

# ----------------------------
# Select Dataset (Modify as Needed)
# ----------------------------
TABULAR_DATA="/data/mra_midas/data_1.xlsx" 
IMAGE_FOLDER="/data/mra_midas/images"

if [ "$#" -ge 1 ]; then
    TABULAR_DATA="/data/mra_midas/$1.xlsx" 
fi

# ----------------------------
# Launch PyTorch Distributed Classification
# ----------------------------

echo "Launching PyTorch Distributed Classification on Master Node (namenode)"
echo "Using Tabular Data: $TABULAR_DATA"
echo "Using Image Folder: $IMAGE_FOLDER"

# Launch torchrun with 4 processes (nproc_per_node=4)
torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$PYTORCH_SCRIPT_PATH" \
    --tabular_data "$TABULAR_DATA" \
    --image_data "$IMAGE_FOLDER"

echo "PyTorch torchrun launched on namenode with node_rank=0."
