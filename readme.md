# Deep-Contam Baselines

## Setup

This repository provides baseline implementations for evaluating two model families:

- **llava-v1.5-7b** (following the official LLaVA example workflow)
- **Qwen/Qwen2-VL-7B-Instruct**

### Installation

For **llava-v1.5-7b**, install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

For **Qwen/Qwen2-VL-7B-Instruct**, install the additional utilities:

```bash
pip install qwen-vl-utils
```

## Data Storage

Follow the Hugging Face dataset layout by placing the TSV file and images under:

```
u/j/a/jadenpark/LMUData/images/
```

The directory should contain:

- A `.tsv` data file (e.g., `RealWorldQA_local.tsv`)
- Corresponding image folder (e.g., `RealWorldQA`)

Example structure:

```
u/j/a/jadenpark/LMUData/images/
├── RealWorldQA_local.tsv
├── RealWorldQA── 1.jpg
 ...
```

If you wish to use a different dataset, adjust the data loading and processing logic in the code accordingly.