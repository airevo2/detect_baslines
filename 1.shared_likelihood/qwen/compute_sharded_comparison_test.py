import os
from PIL import Image
import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LlavaForConditionalGeneration

import math
import random 

import numpy as np 
from scipy.stats import binom
from scipy.stats import t as tdist

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import GPUtil
from multiprocessing import Process, Queue

from tqdm import tqdm

import json

import fire

import multiprocessing
# For supporting CUDA in subprocesses, force using spawn start method
multiprocessing.set_start_method('spawn', force=True)

os.environ['TOKENIZERS_PARALLELISM'] = "True"

flatten = lambda l : [x for s in l for x in s]
shuffle = lambda l : random.sample(l, k=len(l))

def load_dataset(dataset_path):
    # For loading a JSON-serialized list of examples.
    if dataset_path.endswith(".json"):
        print("loading from json...")
        with open(dataset_path, "r") as f:
            data = f.read()
            examples = json.loads(data)
            return examples

    # For loading a dataset where each example is on its own line.
    with open(dataset_path, "r") as f:
        lines = f.readlines()
    return lines

def compute_logprob_of_token_sequence(tokens, model, context_len=2048, stride=1024, device=0):
  # Support multimodal BatchFeature or dict type input
  if hasattr(tokens, 'pixel_values') or isinstance(tokens, dict):
      return compute_logprob_of_encoding(tokens, model, context_len, stride, device)
  # Original logic for text sequence
  """Approximates logp(tokens) by sliding a window over the tokens with a stride."""
  inputs  = tokens[:-1]
  targets = tokens[1:]

  logp = torch.zeros((1, 1), dtype=torch.float32).to(device)

  # compute the smallest multiple k of s so that t <= ks + c.
  t = len(inputs); c = context_len; s = stride
  k = math.ceil(max(0, t - c) / s)
  all_logps = []
  for j in range(k + 1):
    start    = s * j
    end      = min(s * j + c, t)
    rel_offs = max(0, c - s) if j > 0 else 0

    w_inp = inputs[start:end]; w_inp = torch.tensor(w_inp).to(device)
    w_trg = targets[start:end]; w_trg = torch.tensor(w_trg).to(device)

    model.eval()
    with torch.no_grad():
      out = model(torch.unsqueeze(w_inp, 0))
      logps = torch.nn.functional.log_softmax(out.logits[0], dim=-1)
      logps = logps.gather(-1, w_trg.unsqueeze(-1)).squeeze(-1)
      logp += logps[rel_offs:].sum()

    del w_inp
    del w_trg
    torch.cuda.empty_cache()

  return logp.item()

def compute_logprob_of_encoding(encoding, model, context_len=2048, stride=1024, device=0):
  """
  Approximates logp(text | image) by sliding a window over the combined text tokens, passing pixel_values each time.
  """
  # Single sample batch
  input_ids = encoding['input_ids'][0]        # shape [seq_len]
  attention_mask = encoding['attention_mask'][0]
  pixel_values = encoding['pixel_values'].to(device)
  # Get image_grid_thw and move to device
  grid_thw = encoding.get('image_grid_thw', None)
  if grid_thw is not None:
    grid_thw = grid_thw.to(device)

  total_logp = 0.0
  t = input_ids.size(0)
  c, s = context_len, stride
  k = math.ceil(max(0, t - c) / s)
  for j in range(k + 1):
    start = s * j
    end = min(s * j + c, t)
    rel_offs = max(0, c - s) if j > 0 else 0

    w_ids = input_ids[start:end].unsqueeze(0).to(device)
    w_mask = attention_mask[start:end].unsqueeze(0).to(device)
    # Construct model input, containing visual and text
    inputs = {
      'input_ids': w_ids,
      'attention_mask': w_mask,
      'pixel_values': pixel_values
    }
    if grid_thw is not None:
      inputs['image_grid_thw'] = grid_thw
    with torch.no_grad():
      out = model(**inputs)
      logits = out.logits  # [batch, seq_len, vocab]
    # shift for next-token logp
    shift_logits = logits[..., :-1, :]
    shift_labels = w_ids[..., 1:].unsqueeze(-1)
    logps = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    # gather token logp
    token_logp = logps.gather(-1, shift_labels).squeeze(-1)
    # mask
    shift_mask = w_mask[..., 1:]
    # accumulate only valid positions
    total_logp += (token_logp * shift_mask).sum().item()
  return total_logp

def worker(model_name_or_path,
           context_len,
           stride,
           device,
           main_queue,
           worker_queue,
           model_type='text'):
    
        #test
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    
    # Load model based on type
    if model_type == 'multimodal':
        m = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype='auto', device_map={'': device}, trust_remote_code=True
        )
    elif model_type == 'llava':
        m = LlavaForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    else:
        m = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    m.cuda(device)
    main_queue.put((device, True))
    
    # Wait for inference requests.
    while True:
        tokens, shard_id, is_canonical = worker_queue.get()

        if tokens == None: # Quit.
            break

        # Compute logprob of tokens.
        logprob = compute_logprob_of_token_sequence(tokens, 
                                                    m, 
                                                    context_len, 
                                                    stride,
                                                    device=device)

        # Send result to main process.
        main_queue.put((logprob, shard_id, is_canonical))
        
    del m

def main(model_name_or_path,
         dataset_path,
         model_type='text',
         dataset_format='json',
         context_len=2048,
         stride=1024,
         num_shards=50,
         permutations_per_shard=250,
         random_seed=0,
         log_file_path=None,
         max_examples=100,  # Run only first ten examples for validation
         m=10,
         k=10):

    # Set random seed(s).
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Load the dataset.
    if dataset_format == 'tsv':
        # Local implementation of RealWorldQA loader
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
    
    # Load model and processor for multimodal or Llava
    if model_type in ('multimodal', 'llava'):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if model_type == 'multimodal':
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype='auto', device_map='auto', trust_remote_code=True
            )
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        model = model.half().to(device)
        model.eval()
    # Text scenario: load model inside worker; no tokenizer needed here

    # Tokenize or prepare examples
    if model_type in ('multimodal', 'llava') and dataset_format == 'tsv':
        encodings = []
        for item in examples:
            img_path = item['image_path'].lstrip('/')
            img = Image.open(img_path).convert('RGB')
            # Construct messages and generate text with image_token placeholder
            prompt_text = item.get('question', '')
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': img_path},
                    {'type': 'text', 'text': prompt_text}
                ]
            }]
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Joint image-text encoding
            encoding = processor(
                text=[text_input],
                images=[img],
                return_tensors='pt',
                padding=True
            )
            encodings.append(encoding)
        tokenized_examples = encodings
        # No random permutations in multimodal scenario
        permutations_per_shard = 10
        num_shards = len(tokenized_examples)
    else:
        t = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenized_examples = [t.encode(ex) for ex in examples]

    # Launch a Process for each GPU.
    gpus = GPUtil.getGPUs()
    print("here is the gpu", [i.id for i in gpus])
    #gpus = [gpu for gpu in gpus if gpu.id != '1']
    gpu_indices = [0]
    gpus = [gpus[i] for i in gpu_indices]
    num_workers = len(gpus)
    processes = []
    main_queue = Queue()
    worker_queues = [Queue() for _ in range(num_workers)]
    print("here is the gpu in use", [i.id for i in gpus])
    for i, gpu in enumerate(gpus):
        p = Process(target=worker, args=(model_name_or_path,
                                         context_len,
                                         stride,
                                         gpu.id,
                                         main_queue,
                                         worker_queues[i],
                                         model_type))
        processes.append(p)
        p.start()
        
    # Wait until each GPU has loaded a model.
    num_ready = 0
    while num_ready < num_workers:
        gpu_id, is_ready = main_queue.get()
        print(f"GPU {gpu_id} loaded model.")
        num_ready += 1
    
    # Issue requests to all worker queues, round-robin style.
    
    # Compute the number of examples for each shard.
    shard_counts = [(x + 1 if i < num_examples % num_shards else x) 
       for i, x in enumerate([num_examples // num_shards] * num_shards)]
    shard_counts = np.asarray(shard_counts)

    # Dispatch tasks to workers
    if model_type in ('multimodal', 'llava'):
        # initialize a tokenizer for subtoken shuffling
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print(f"[DEBUG] Starting dispatch: {len(examples)} examples, {permutations_per_shard} shuffles each.")
        # For each example, dispatch canonical encoding and shuffled-text encodings
        for i, item in enumerate(examples):
            print(f"[DEBUG] Example {i}: dispatching canonical encoding.")
            # canonical (precomputed) encoding
            encoding = tokenized_examples[i]
            worker_queues[0].put((encoding, i, True))
            print(f"[DEBUG] Example {i}: dispatching shuffled encodings.")
            # prepare image and original text
            img_path = item['image_path'].lstrip('/')
            img = Image.open(img_path).convert('RGB')
            # prepare question and answer options, then shuffle options and re-encode
            question = item.get('question', '')
            # Collect answer options from dataset columns
            options = [item[k] for k in ['A', 'B', 'C', 'D', 'E'] if k in item]
            for j in range(permutations_per_shard):
                w = j % num_workers
                print(f"[DEBUG] Example {i}, shuffle {j}: sending to worker {w}.")
                # Shuffle only the answer options
                shuffled_opts = random.sample(options, k=len(options))
                # Build content list: image, question, then shuffled option texts
                content = [{'type': 'image', 'image': img_path},
                           {'type': 'text', 'text': question}]
                for opt in shuffled_opts:
                    content.append({'type': 'text', 'text': opt})
                messages = [{'role': 'user', 'content': content}]
                text_input = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                new_encoding = processor(
                    text=[text_input],
                    images=[img],
                    return_tensors='pt',
                    padding=True
                )
                worker_queues[w].put((new_encoding, i, False))
    else:
        # Compute the starting index (into the list of examples) for each shard.
        shard_example_indices = [0] + np.cumsum(shard_counts).tolist()
        for i, (start, end) in enumerate(zip(shard_example_indices, shard_example_indices[1:])):
            shard = tokenized_examples[start:end]
            # Logprobs in canonical order.
            worker_queues[0].put((
                flatten(shard), # tokens
                i,              # shard id
                True))          # is_canonical=True
            # Logprobs in shuffled order(s). 
            for j in range(permutations_per_shard):
                w = j % num_workers
                worker_queues[w].put((
                    flatten(shuffle(shard)), # tokens
                    i,                       # shard id
                    False))                  # is_canonical=False

    # Wait on requests.
    total_work = num_shards * (1 + permutations_per_shard)
    pbar = tqdm(total=total_work)

    canonical_logprobs = [None for _ in range(num_shards)]
    shuffled_logprobs  = [[] for _ in range(num_shards)]

    completed = 0
    while completed < total_work:
        
        logprob, shard_id, is_canonical = main_queue.get()

        if is_canonical:
            canonical_logprobs[shard_id] = logprob 
        else:
            shuffled_logprobs[shard_id].append(logprob)
            
        pbar.update(1)
        completed += 1

    # Terminate workers.
    for w in range(num_workers):
        worker_queues[w].put((None, None, None))

    for p in processes:
        p.join()

    # Calculate p-value.
    canonical_logprobs = np.asarray(canonical_logprobs)
    shuffled_logprobs  = np.asarray(shuffled_logprobs)
    
    # Subsample p-value estimation.
    diffs = canonical_logprobs - shuffled_logprobs.mean(axis=1)
    pval = subsample_pval(diffs, m=m, K=k)
    print(pval)

    # Log.
    if log_file_path is not None:
        print(f"Writing logprobs to: {log_file_path}")
        with open(f"{log_file_path}", 'w') as f:
            f.write(json.dumps({
                'pval': pval, 
                'permutations_per_shard': permutations_per_shard,
                'num_shards': num_shards,
                'canonical_logprobs': canonical_logprobs.tolist(),
                'shuffled_logprobs': shuffled_logprobs.tolist(),
            }))

def subsample_pval(diffs, m=5, K=20):
    ps = []
    for _ in range(K):
        sub = np.random.choice(diffs, m, replace=False)
        z = sub.mean() / (sub.std()+0.0001) * np.sqrt(m)
        ps.append(1 - tdist.cdf(z, df=m-1))
    return np.median(ps)

if __name__ == '__main__':
  fire.Fire(main)
