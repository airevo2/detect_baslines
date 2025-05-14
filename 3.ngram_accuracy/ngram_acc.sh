export CUDA_VISIBLE_DEVICES=0
python ngram_acc.py \
  --dataset_name u/j/a/jadenpark/LMUData/images/RealWorldQA_local.tsv \
  --model_path jpark677/qwen2-vl-7b-instruct-realworldqa-fft-unfreeze-all-ep-3-waa-f \
  --model_name qwen2vl \
  --model_type multimodal \
  --device cuda:0 \
  --n 5

    # --num_samples 50 \