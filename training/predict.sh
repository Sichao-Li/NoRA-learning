#!/usr/bin/env bash
# Use the same inputs and generation options for RAW and SFT checkpoints.
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: bash training/predict.sh MODEL TEST_JSONL OUTPUT_JSONL [swift options...]" >&2
    exit 2
fi
model=$1
data=$2
output=$3
shift 3
[[ -s "$data" ]] || { echo "Missing test inputs: $data" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 2; }
mkdir -p "$(dirname "$output")"

export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-1024}
export TOKENIZERS_PARALLELISM=false

swift infer \
    --model "$model" \
    --val_dataset "$data" \
    --result_path "$output" \
    --write_batch_size 1 \
    --max_batch_size 1 \
    --max_length 4096 \
    --max_new_tokens 2048 \
    --temperature 0 \
    --top_k 1 \
    --enable_thinking false \
    --remove_unused_columns true \
    --torch_dtype bfloat16 \
    --attn_impl sdpa "$@"
