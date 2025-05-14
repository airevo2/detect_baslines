import os
from PIL import Image
import argparse
import math, random
import numpy as np
from scipy.stats import t as tdist
import torch
from transformers import AutoTokenizer
import GPUtil
from multiprocessing import Process, Queue
from tqdm import tqdm
import json
import fire
import multiprocessing
# Use LLaVA builder API to load model and processor
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.mm_utils import process_images, tokenizer_image_token
from llava.conversation import conv_templates

# For supporting CUDA in subprocesses, force using spawn start method
multiprocessing.set_start_method('spawn', force=True)
os.environ['TOKENIZERS_PARALLELISM'] = "True"

# Utility functions
def flatten(l): return [x for s in l for x in s]
def shuffle(l): return random.sample(l, k=len(l))

# Load dataset
def load_dataset(dataset_path):
    if dataset_path.endswith(".json"):
        with open(dataset_path, "r") as f:
            return json.load(f)
    with open(dataset_path, "r") as f:
        return [line.strip() for line in f]

# Compute logprob for token sequence
def compute_logprob_of_token_sequence(tokens, model, context_len=2048, stride=1024, device=0):
    if hasattr(tokens, 'pixel_values') or isinstance(tokens, dict):
        return compute_logprob_of_encoding(tokens, model, context_len, stride, device)
    inputs = tokens[:-1]
    targets = tokens[1:]
    logp = torch.zeros((1,1), dtype=torch.float32).to(device)
    t_len, c, s = len(inputs), context_len, stride
    k = math.ceil(max(0, t_len - c) / s)
    for j in range(k+1):
        start = s*j
        end = min(s*j + c, t_len)
        rel_offs = max(0, c - s) if j>0 else 0
        w_inp = torch.tensor(inputs[start:end]).to(device)
        w_trg = torch.tensor(targets[start:end]).to(device)
        with torch.no_grad():
            out = model(torch.unsqueeze(w_inp, 0))
            logps = torch.nn.functional.log_softmax(out.logits[0], dim=-1)
            logps = logps.gather(-1, w_trg.unsqueeze(-1)).squeeze(-1)
            logp += logps[rel_offs:].sum()
        del w_inp, w_trg
        torch.cuda.empty_cache()
    return logp.item()

def compute_logprob_of_encoding(encoding, model, context_len=2048, stride=1024, device=0):
    # Extract multimodal inputs
    input_ids = encoding['input_ids'][0]
    attention_mask = encoding['attention_mask'][0]
    images = encoding['pixel_values'].to(device)
    image_sizes = encoding.get('image_sizes', None)
    total_logp = 0.0
    t, c, s = input_ids.size(0), context_len, stride
    k = math.ceil(max(0, t - c) / s)
    for j in range(k+1):
        start = s*j
        end = min(s*j + c, t)
        rel_offs = max(0, c - s) if j>0 else 0
        w_ids = input_ids[start:end].unsqueeze(0).to(device)
        w_mask = attention_mask[start:end].unsqueeze(0).to(device)
        # Prepare inputs for LLaVA forward: images & image_sizes
        inputs = {'input_ids': w_ids, 'attention_mask': w_mask, 'images': images}
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits
        # shift for next-token logp
        shift_logits = logits[..., :-1, :]
        shift_labels = w_ids[..., 1:].unsqueeze(-1)
        logps = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_logp = logps.gather(-1, shift_labels).squeeze(-1)
        shift_mask = w_mask[..., 1:]
        total_logp += (token_logp * shift_mask).sum().item()
    return total_logp

# Worker process that loads LLaVA model and computes logprobs
def worker(model_name_or_path, context_len, stride, device, main_queue, worker_queue):
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    # Load LLaVA model via builder API
    model_name = get_model_name_from_path(model_name_or_path)
    tokenizer, m, image_processor, llava_context_len = load_pretrained_model(
        model_path=model_name_or_path,
        model_base=None,
        model_name=model_name
    )
    m.cuda(device)
    main_queue.put((device, True))
    while True:
        tokens, shard_id, is_canonical = worker_queue.get()
        if tokens is None:
            break
        logprob = compute_logprob_of_token_sequence(tokens, m, context_len, stride, device=device)
        main_queue.put((logprob, shard_id, is_canonical))
    del m

# Main function specialized for LLaVA
def main(model_name_or_path,
         dataset_path,
         dataset_format='tsv',
         context_len=2048,
         stride=1024,
         num_shards=50,
         permutations_per_shard=250,
         random_seed=0,
         log_file_path=None,
         max_examples=100,
         m=10,
         k=10):
    random.seed(random_seed)
    np.random.seed(random_seed)
    # Load examples
    if dataset_format == 'tsv':
        examples = []
        import csv
        with open(dataset_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for item in reader:
                examples.append(item)
    else:
        examples = load_dataset(dataset_path)
    examples = examples[:max_examples]
    num_examples = len(examples)
    print(f"Loaded {num_examples} examples from {dataset_path}")
    # Load LLaVA model, tokenizer and processor via builder API
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_name = get_model_name_from_path(model_name_or_path)
    tokenizer, model, image_processor, llava_context_len = load_pretrained_model(
        model_path=model_name_or_path,
        model_base=None,
        model_name=model_name
    )
    model = model.half().to(device)
    model.eval()
    # Prepare multimodal encodings using LLaVA 官方流程
    encodings = []
    for item in examples:
        # construct prompt: use <image> placeholder, tokenizer_image_token will replace it with the correct image token
        qs = f"{IMAGE_PLACEHOLDER}\n{item['question']}"
        conv = conv_templates['llava_v1'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # process image
        img = Image.open(item['image_path'].lstrip('/')).convert('RGB')
        pixel_values = process_images([img], image_processor, model.config).to(device, torch.float16)
        # Tokenize prompt: get sequence, then add batch dim
        ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = ids.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        encodings.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'pixel_values': pixel_values, 'image_sizes': [img.size]})
    tokenized_examples = encodings
    permutations_per_shard = permutations_per_shard
    num_shards = len(tokenized_examples)
    # Launch processes
    gpus = GPUtil.getGPUs()
    gpu_indices = [0]
    gpus = [gpus[i] for i in gpu_indices]
    num_workers = len(gpus)
    processes = []
    main_queue = Queue()
    worker_queues = [Queue() for _ in range(num_workers)]
    for i, gpu in enumerate(gpus):
        p = Process(target=worker, args=(model_name_or_path, context_len, stride, gpu.id, main_queue, worker_queues[i]))
        processes.append(p)
        p.start()
    # Wait for model loading
    num_ready = 0
    while num_ready < num_workers:
        gpu_id, ready = main_queue.get()
        print(f"GPU {gpu_id} loaded model.")
        num_ready += 1
    # Dispatch tasks using LLaVA 官方推理流程
    total_work = num_shards * (1 + permutations_per_shard)
    # Debug: starting dispatch of canonical and shuffled encodings
    print(f"[DEBUG] Starting dispatch: {len(examples)} examples, {permutations_per_shard} shuffles each.")
    for i, enc in enumerate(tokenized_examples):
        # Debug: dispatching canonical encoding for example i
        print(f"[DEBUG] Example {i}: dispatching canonical encoding.")
        # canonical (precomputed) encoding
        worker_queues[0].put((enc, i, True))
        # Debug: dispatching shuffled encodings for example i
        print(f"[DEBUG] Example {i}: dispatching shuffled encodings.")
        # shuffled permutations
        item = examples[i]
        options = [item[k] for k in ['A','B','C','D'] if k in item]
        for j in range(permutations_per_shard):
            w = j % num_workers
            # Debug: shuffle j sent to worker w
            print(f"[DEBUG] Example {i}, shuffle {j}: sending to worker {w}.")
            shuffled_opts = random.sample(options, k=len(options))
            # construct and encode shuffled prompt
            qs2 = f"{IMAGE_PLACEHOLDER}\n{item['question']}\nOptions:\n" + "\n".join(shuffled_opts)
            conv2 = conv_templates['llava_v1'].copy()
            conv2.append_message(conv2.roles[0], qs2)
            prompt2 = conv2.get_prompt()
            # image processing
            img2 = Image.open(item['image_path'].lstrip('/')).convert('RGB')
            pixel_values2 = process_images([img2], image_processor, model.config).to(device, torch.float16)
            # text encoding
            ids2 = tokenizer_image_token(prompt2, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids2 = ids2.unsqueeze(0).to(device)
            attention_mask2 = torch.ones_like(input_ids2)
            new_enc = {
                'input_ids': input_ids2,
                'attention_mask': attention_mask2,
                'pixel_values': pixel_values2,
                'image_sizes': [img2.size]
            }
            worker_queues[w].put((new_enc, i, False))
    # Collect results
    canonical_logprobs = [None] * num_shards
    shuffled_logprobs = [[] for _ in range(num_shards)]
    pbar = tqdm(total=total_work)
    completed = 0
    while completed < total_work:
        logprob, shard_id, is_canonical = main_queue.get()
        if is_canonical:
            canonical_logprobs[shard_id] = logprob
        else:
            shuffled_logprobs[shard_id].append(logprob)
        completed += 1
        pbar.update(1)
    # Terminate workers
    for w in range(num_workers):
        worker_queues[w].put((None, None, None))
    for p in processes:
        p.join()
    # Compute p-value
    canonical_logprobs = np.asarray(canonical_logprobs)
    shuffled_logprobs = np.asarray(shuffled_logprobs)
    diffs = canonical_logprobs - shuffled_logprobs.mean(axis=1)
    pval = subsample_pval(diffs, m=m, K=k)
    print(pval)
    if log_file_path is not None:
        with open(log_file_path, 'w') as f:
            f.write(json.dumps({'pval': pval, 'differences': diffs.tolist()}))

def subsample_pval(diffs, m=5, K=20):
    ps = []
    for _ in range(K):
        sub = np.random.choice(diffs, m, replace=False)
        z = sub.mean() / (sub.std() + 1e-4) * math.sqrt(m)
        ps.append(1 - tdist.cdf(z, df=m-1))
    return np.median(ps)

if __name__ == '__main__':
    fire.Fire(main)
