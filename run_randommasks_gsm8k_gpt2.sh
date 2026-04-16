#!/bin/bash
export PYTHONPATH="/home/ss19021/Internalize_CoT_Step_by_Step:$PYTHONPATH"

cd /home/ss19021/Internalize_CoT_Step_by_Step/ || exit

SCRATCH_DIR="/scratch/ss19021/Internalize_CoT_Step_by_Step"
DIR="$SCRATCH_DIR/train_models/gsm8k/gpt2/gpt2_randommasks/"
mkdir -p "$DIR"

which python
which accelerate

accelerate launch \
  --mixed_precision=bf16 \
  --num_processes=${SLURM_GPUS_ON_NODE} \
  --num_machines=1 \
src/train.py \
    --mode train \
    --model gpt2 \
    --huggingface_dataset \
    --path openai/gsm8k \
    --name main \
    --train_split train \
    --val_split "test;val" \
    --test_split "test;test" \
    --best_val_metric "full-cot_accuracy" \
    --epochs 96 \
    --lr 5e-5 \
    --batch_size 8 \
    --accumulate 16 \
    --n_generative_eval_batches 64 \
    --intermediate_eval \
    --train_type cot-distill \
    --removal_type random-masks \
    --joint_masked_distribution \
    --replace_mask \
    --prompt_in_percentage \
    --pretrain_epochs 0 \
    --eval_period 3 \
    --save_period 30 \
    --seed 3456 \
    --reset_optimizer \
    --wandb_project cot-distillation \
    --wandb_run_name b8_gsm8k_randommasks_gpt2 \
    --max_new_tokens 512 \
    --save_model "$DIR" \
    > "$DIR/log_masks.train" 2>&1
