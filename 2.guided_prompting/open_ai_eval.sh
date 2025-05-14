#!/usr/bin/env bash

# Set GPU device
CUDA_VISIBLE_DEVICES=1 \
  python3 open_ai_eval.py \
  --dataset_path u/j/a/jadenpark/LMUData/images/RealWorldQA_local.tsv \
  --model_path Qwen/Qwen2-VL-7B-Instruct \
  --device cuda:0 \
  --output vision_eval_results_qwen.txt \
  --max_new_tokens 256 \
  --openai_api_key "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" \
  --openai_model gpt-4o