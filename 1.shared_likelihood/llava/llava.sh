export CUDA_VISIBLE_DEVICES=0
   python compute_sharded_comparison_test_llava.py \
     --dataset_path u/j/a/jadenpark/LMUData/images/RealWorldQA_local.tsv \
     --model_name_or_path liuhaotian/llava-v1.5-7b \
     --dataset_format tsv \
     --context_len 4096 \
     --stride 1024 \
     --num_shards 50 \
     --permutations_per_shard 10 \
     --log_file_path llava_realworldqa_shards_new.log \
     --m 4 \
     --k 10