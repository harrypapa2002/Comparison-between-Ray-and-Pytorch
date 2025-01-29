#!/bin/bash

# ============================================================
# Launch Script for Ray Distributed Lesion Classification
# ============================================================

# ----------------------------
# Default Configuration
# ----------------------------
DEFAULT_DATASET="data_1" 

# ----------------------------
# Parse Command-Line Arguments
# ----------------------------
# Usage: ./launch_ray.sh <DATASET>
if [ "$#" -ge 1 ]; then
    DATASET=$1 
else
    DATASET=$DEFAULT_DATASET  

# ----------------------------
# Ray Python Script Path
# ----------------------------
RAY_SCRIPT_PATH="/home/ubuntu/Comparison-between-Ray-and-Pytorch/LesionClassification/lesion_classification_ray.py"

# ----------------------------
# Print Configuration
# ----------------------------
echo "Launching Ray Distributed Classification"
echo "Using Dataset: $DATASET"

# ----------------------------
# Launch Ray Distributed Training
# ----------------------------
python "$RAY_SCRIPT_PATH" --dataset "$DATASET"

echo "Ray training launched with dataset=$DATASET."
