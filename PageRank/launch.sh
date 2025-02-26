#!/bin/bash

# ============================================================
# Launch Script for Distributed PyTorch PageRank
# Master Node: namenode
# Processes: 4 (one per CPU core)
# ============================================================

# ----------------------------
# Configuration Parameters
# ----------------------------

# Master Node Details
MASTER_ADDR="namenode"
MASTER_PORT=29500  # Keep this consistent with your PageRank code
NNODES=1 # Number of worker nodes
NPROC_PER_NODE=4  # Number of processes per node (adjust to your CPU cores)
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))  # Total number of processes

# HDFS Configuration
HDFS_HOST="namenode"
HDFS_PORT=9000
DATA_FILE="/data/twitter7/twitter7_5gb.csv"  # Path to your HDFS data

# PageRank Parameters
BATCH_SIZE=$((1024 * 1024 * 30))  
OUTPUT_FILE="pytorch_pagerank_node3_results.json"

# PyTorch PageRank Script Path
PYTORCH_SCRIPT_PATH="/home/ubuntu/Comparison-between-Ray-and-Pytorch/PageRank/pagerank.py"

# Log Directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ----------------------------
# Launch PyTorch Distributed PageRank
# ----------------------------

echo "Launching PyTorch Distributed PageRank on Master Node (namenode)"

torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$PYTORCH_SCRIPT_PATH" \
    --datafile "$DATA_FILE" \
    --batch_size "$BATCH_SIZE" \
    --hdfs_host "$HDFS_HOST" \
    --hdfs_port "$HDFS_PORT" \
    > "$LOG_DIR/log_pagerank_master.txt" 2>&1 &

echo "PyTorch torchrun launched for PageRank on namenode with node_rank=0."

