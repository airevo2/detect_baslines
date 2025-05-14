export CUDA_VISIBLE_DEVICES=1
   python compute_sharded_comparison_test.py \
     --dataset_path u/j/a/jadenpark/LMUData/images/RealWorldQA_local.tsv \
     --model_name_or_path jpark677/qwen2-vl-7b-instruct-realworldqa-fft-unfreeze-all-ep-3-waa-f \
     --model_type multimodal \
     --dataset_format tsv \
     --context_len 4096 \
     --stride 1024 \
     --num_shards 50 \
     --permutations_per_shard 500 \
     --log_file_path qwen_contaminated_realworldqa_shards.log \
     --m 4 \
     --k 100