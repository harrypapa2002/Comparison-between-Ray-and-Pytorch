#!/bin/bash

# ============================================================
# Launch Script for PyTorch Distributed KMeans Clustering
# Master Node: namenode
# Processes: 4 (one per CPU core)
# ============================================================

# ----------------------------
# Configuration Parameters
# ----------------------------

# Master Node Details
MASTER_ADDR="namenode"
MASTER_PORT=29500
NNODES=3
NPROC_PER_NODE=4
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))  # 12

# HDFS Configuration
HDFS_HOST="namenode"
HDFS_PORT=9000
FILE_PATH="/data/nyc_taxi/yellow_tripdata_2015-01.csv"

# Clustering Parameters
N_CLUSTERS=40
READ_BLOCK_SIZE=1048576  # 1 MB
DATA_LOADER_BATCH_SIZE=3340  # Number of samples per batch

# Output File
OUTPUT_FILE="pytorch_data1_node3.json"

# PyTorch Script Path (absolute path)
PYTORCH_SCRIPT_PATH="/home/ubuntu/Comparison-between-Ray-and-Pytorch/NYC_YellowTaxiTrip/nyc_taxi_pytorch_cluster.py"

# Log Directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ----------------------------
# Launch PyTorch Distributed Training
# ----------------------------

echo "Launching PyTorch Distributed Training on Master Node (namenode)"

# Launch torchrun with 4 processes (nproc_per_node=4)
torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$PYTORCH_SCRIPT_PATH" \
    --files $FILE_PATH \
    --hdfs_host "$HDFS_HOST" \
    --hdfs_port "$HDFS_PORT" \
    --read_block_size "$READ_BLOCK_SIZE" \
    --data_loader_batch_size "$DATA_LOADER_BATCH_SIZE" \
    --n_clusters "$N_CLUSTERS" \
    --output "$OUTPUT_FILE" \
    > "$LOG_DIR/log_master_torchrun.txt" 2>&1 &

echo "PyTorch torchrun launched on namenode with node_rank=0."
