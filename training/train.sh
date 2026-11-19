#!/usr/bin/env bash
# Resource-independent form of the MS-SWIFT full-SFT command.
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: bash training/train.sh MODEL PREPARED_DATA OUTPUT [swift options...]" >&2
    exit 2
fi
model=$1
data=$2
output=$3
shift 3
[[ -s "$data/train.jsonl" && -s "$data/val.jsonl" ]] || { echo "Missing prepared training data" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 2; }

export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-1024}
export TOKENIZERS_PARALLELISM=false

swift sft \
    --model "$model" \
    --tuner_type full \
    --dataset "$data/train.jsonl" \
    --val_dataset "$data/val.jsonl" \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --max_length 4096 \
    --gradient_checkpointing true \
    --group_by_length true \
    --loss_scale default+ignore_empty_think \
    --add_non_thinking_prefix true \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --save_steps 50 \
    --eval_steps 50 \
    --logging_steps 5 \
    --save_total_limit 2 \
    --output_dir "$output" "$@"
