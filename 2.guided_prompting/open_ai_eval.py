import argparse
import openai
import csv, random, os
from PIL import Image
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def get_usage():
    try:
        usage = openai.Usage.retrieve()
        print(f"Usage Information: {usage}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_billing_info():
    try:
        billing_info = openai.Billing.retrieve()
        print(f"Billing Information: {billing_info}")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_realworldqa(path):
    ds = {"image_path": [], "question": [], "options": [], "answer": []}
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for item in reader:
            img = item.get('image_path', '').lstrip('/')
            ds['image_path'].append(img)
            ds['question'].append(item.get('question', ''))
            opts = {k: item[k] for k in ['A','B','C','D'] if k in item and item[k]}
            ds['options'].append(opts)
            ds['answer'].append(item.get('answer', ''))
    return ds

def main():
    parser = argparse.ArgumentParser(description='Vision eval with Qwen2-VL on RealWorldQA')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to RealWorldQA TSV file')
    parser.add_argument('--model_path', type=str, required=True, help='Qwen2-VL model path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Compute device')
    parser.add_argument('--output', type=str, default='vision_eval_results.txt', help='Output file')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Max new tokens')
    parser.add_argument('--openai_api_key', type=str, required=True, help='OpenAI API key for semantic check')
    parser.add_argument('--openai_model', type=str, default='gpt-4o', help='OpenAI model for semantic check')
    args = parser.parse_args()
    openai.api_key = args.openai_api_key

    # Load Qwen2VL model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype='auto', device_map='auto', trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval().half().to(args.device)

    # Load dataset
    ds = load_realworldqa(args.dataset_path)
    results = []

    for img_path, question, opts_dict, true_label in tqdm(zip(ds['image_path'], ds['question'], ds['options'], ds['answer']), total=len(ds['question'])):
        # mask one option
        masked_label = random.choice(list(opts_dict.keys()))
        ground_truth = opts_dict[masked_label]
        remaining_texts = [txt for lbl, txt in opts_dict.items() if lbl != masked_label]
        display_opts = "\n".join(f"- {txt}" for txt in remaining_texts)
        prompt_text = (
            f"{question}\n"
            "Here are some answer choices (one is missing):\n"
            f"{display_opts}\n"
            "Please predict the missing answer choice text only, without any labels."
        )
        # construct messages and template
        messages = [
            {"role":"user","content":[{"type":"image","image":img_path},{"type":"text","text":prompt_text}]} 
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # process image
        img = Image.open(img_path).convert('RGB')
        encoding = processor(
            text=[text],
            images=[img],
            return_tensors='pt',
            padding=True
        )
        encoding = {k: v.to(args.device) for k, v in encoding.items()}
        # generate
        outputs = model.generate(**encoding, max_new_tokens=args.max_new_tokens)
        generated = [out[len(inp):] for inp, out in zip(encoding['input_ids'], outputs)]
        pred = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        # semantic equivalence detection
        sem_prompt = f"""Compare the semantics of two texts. Answer 'true' if they have the same meaning, 'false' otherwise. No explanations.

Example:
Text1: How are you?
Text2: How are you doing?
Answer: true

Text1: The sky is blue.
Text2: The sky is clear.
Answer: false

Text1: {pred}
Text2: {ground_truth}
Answer:"""
        response = openai.chat.completions.create(model=args.openai_model, messages=[{'role':'user','content':sem_prompt}])
        equiv = response.choices[0].message.content.strip().lower()
        results.append((masked_label, ground_truth, pred, equiv))

    # write results
    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    with open(args.output, 'w', encoding='utf-8') as f:
        for lbl, truth, pred, equiv in results:
            f.write(f"{lbl}\t{truth}\t{pred}\t{equiv}\n")

    total = len(results)
    correct = sum(1 for item in results if item[3] == 'true')
    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"Semantic equivalence accuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == '__main__':
    main()
