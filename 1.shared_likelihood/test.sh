export CUDA_VISIBLE_DEVICES=0

python original.py Qwen/Qwen1.5-7B-Chat /lustrefs/users/shibo.hao/data/feng/code/junxia/Deep-Contam/detect_baselines/1.shared_likelihood/dataset/math.jsonl --context_len 4096 --stride 1024 --num_shards 50 --permutations_per_shard 10 --log_file_path "qwen_english_4096_50_shards_500_perms_512_mmlu_100.log"