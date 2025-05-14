# Shared_Likelihood

This repository implements the Shared_Likelihood method as detailed in the following repository:

https://github.com/tatsu-lab/test_set_contamination

## Instructions
To evaluate the shared_likelihood of qwen2vl or llava-v1.5 model:

qwen2 vl: 
cd qwen
source qwen.sh
llava:
cd llava
source llava.sh

You can modify the model_path arg in qwen.sh and llava.sh to change the model you need to eval.
As for the dataset, you may need to change the code in compute_sharded_comparison_test.py for data process, it won't be difficult.

