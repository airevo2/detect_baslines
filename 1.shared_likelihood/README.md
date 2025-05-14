# Shared Likelihood Baseline

This repository provides an implementation of the Shared Likelihood method, as described in:

https://github.com/tatsu-lab/test_set_contamination

## Setup

No additional installation is required beyond the model dependencies. To prepare your environment:

```bash
# (Optional) create and activate a virtual environment
eval "$(conda shell.bash hook)"
conda create -n shared_likelihood python=3.10 -y
conda activate shared_likelihood

# Install dependencies
pip install -r requirements.txt
```

## Usage

To evaluate Shared Likelihood on a model, use one of the provided scripts:

### Qwen2VL

```bash
cd qwen
bash qwen.sh --model_path <your_qwen_model_path> --dataset_path <your_dataset>
```

### LLaVA-v1.5-7b

```bash
cd llava
bash llava.sh --model_path <your_llava_model_path> --dataset_path <your_dataset>
```

You can override `model_path` and `dataset_path` arguments in each shell script to point to the desired model checkpoint and data file.

## Data Processing

If you need to use a different dataset format, update the data loading logic in `compute_sharded_comparison_test.py` accordingly. The code is modular and should be straightforward to adapt.

