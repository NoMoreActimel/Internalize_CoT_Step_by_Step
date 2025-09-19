#!/bin/bash

cd /home/ss19021/Internalize_CoT_Step_by_Step/ || exit

SCRATCH_DIR="/scratch/ss19021/Internalize_CoT_Step_by_Step"

OUTPUT_DIR="$SCRATCH_DIR/generated_datasets/shared_graph_cot_${NUM_NODES}n_${NUM_EDGES}e_${NUM_SAMPLES}s/"
mkdir -p "$OUTPUT_DIR"

CANDIDATE_SAMPLES_PATH="$OUTPUT_DIR/candidate_samples.json"
DATASET_PATH="$OUTPUT_DIR/subprosqa_dataset.json"
CONFIG_PATH="$OUTPUT_DIR/config.json"

NUM_NODES=1000
NUM_EDGES=3000
NUM_SAMPLES=1000
NUM_CONTEXT_EDGES=25
REPRESENTATION="structured"
DEPTH_MIN=3
DEPTH_MAX=10
SEED=42
SHOW_SAMPLES=5

python3 src/subprosqa/generate_shared_graph_cot.py \
    --num_nodes "$NUM_NODES" \
    --num_edges "$NUM_EDGES" \
    --num_samples "$NUM_SAMPLES" \
    --num_context_edges "$NUM_CONTEXT_EDGES" \
    --representation "$REPRESENTATION" \
    --depth_range "$DEPTH_MIN" "$DEPTH_MAX" \
    --seed "$SEED" \
    --candidate_samples_path "$CANDIDATE_SAMPLES_PATH" \
    --dataset_path "$DATASET_PATH" \
    --config_path "$CONFIG_PATH" \
    --show_samples "$SHOW_SAMPLES" \
    > "$OUTPUT_DIR/log_generate_graph.txt" 2>&1