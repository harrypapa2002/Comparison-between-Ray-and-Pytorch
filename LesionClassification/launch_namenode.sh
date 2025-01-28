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
NNODES=2
NPROC_PER_NODE=4
WORLD_SIZE=$((NNODES * NPROC_PER_NODE)) 

# PyTorch Script Path (absolute path)
PYTORCH_SCRIPT_PATH="/home/ubuntu/Comparison-between-Ray-and-Pytorch/LesionClassification/lesion_classification_pytorch.py"


# ----------------------------
# Launch PyTorch Distributed Classification
# ----------------------------

echo "Launching PyTorch Distributed Classification on Master Node (namenode)"

# Launch torchrun with 4 processes (nproc_per_node=4)
torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$PYTORCH_SCRIPT_PATH" \

echo "PyTorch torchrun launched on namenode with node_rank=0."
