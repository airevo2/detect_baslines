export CUDA_VISIBLE_DEVICES=0
python ngram_acc_llava.py \
  --dataset_name u/j/a/jadenpark/LMUData/images/RealWorldQA_local.tsv \
  --model_path liuhaotian/llava-v1.5-7b \
  --model_name llava \
  --model_type multimodal \
  --device cuda:0 \
  --n 5

    # --num_samples 50 \