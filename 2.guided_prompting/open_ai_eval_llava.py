import argparse
import openai
import csv, random, os
from PIL import Image
import torch
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates

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
            # Convert to relative image path
            img = item.get('image_path', '').lstrip('/')
            ds['image_path'].append(img)
            ds['question'].append(item.get('question', ''))
            # 保留选项字典
            opts = {k: item[k] for k in ['A','B','C','D'] if k in item and item[k]}
            ds['options'].append(opts)
            ds['answer'].append(item.get('answer', ''))
    return ds

def main():
    parser = argparse.ArgumentParser(description='Vision eval with Qwen2-VL on RealWorldQA')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to RealWorldQA TSV file')
    parser.add_argument('--model_path', type=str, required=True, help='Qwen2-VL model path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Compute device')
    parser.add_argument('--output', type=str, default='vision_eval_results_1.txt', help='Output file')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Max new tokens')
    # OpenAI parameters for semantic equivalence detection
    parser.add_argument('--openai_api_key', type=str, required=True, help='OpenAI API key for semantic check')
    parser.add_argument('--openai_model', type=str, default='gpt-4o', help='OpenAI model for semantic check')
    args = parser.parse_args()
    openai.api_key = args.openai_api_key

    # Load LLaVA model via builder API
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, llava_context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name
    )
    model.eval().half().to(args.device)

    # Load dataset
    ds = load_realworldqa(args.dataset_path)
    # # only use the first 10 samples for testing
    # max_samples = min(10, len(ds['question']))
    # for key in ds:
    #     ds[key] = ds[key][:max_samples]
    # prepare to store results
    results = []
    references = ds['answer']

    # Inference loop
    for img_path, question, opts_dict, true_label in tqdm(zip(ds['image_path'], ds['question'], ds['options'], ds['answer']), total=len(ds['question'])):
        # randomly mask one option (only hide label but keep the true text)
        masked_label = random.choice(list(opts_dict.keys()))
        ground_truth = opts_dict[masked_label]
        # only show the remaining options text, remove labels, encourage model to output complete text
        remaining_texts = [txt for lbl, txt in opts_dict.items() if lbl != masked_label]
        display_opts = "\n".join(f"- {txt}" for txt in remaining_texts)
        prompt_text = (
            f"{question}\n"
            "Here are some answer choices (one is missing):\n"
            f"{display_opts}\n"
            "Please predict the missing answer choice text only, without any labels."
        )
        # Build prompt using <image> placeholder
        qs = f"{IMAGE_PLACEHOLDER}\n{question}\nHere are some answer choices (one is missing):\n" \
             + "\n".join(f"- {txt}" for lbl, txt in opts_dict.items() if lbl != masked_label)
        conv = conv_templates['llava_v1'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # Open and process image
        img = Image.open(img_path).convert('RGB')
        pixel_values = process_images([img], image_processor, model.config).to(args.device, torch.float16)
        # Tokenize prompt and prepare inputs
        ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = ids.unsqueeze(0).to(args.device)
        attention_mask = torch.ones_like(input_ids)
        # Generate outputs with LLaVA
        outputs = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            image_sizes=[img.size],
            max_new_tokens=args.max_new_tokens
        )
        # Decode outputs
        pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
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

    # write the masked option prediction and semantic judgment results
    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    with open(args.output, 'w', encoding='utf-8') as f:
        # format: masked_label, ground_truth, prediction, semantic_equivalent
        for lbl, truth, pred, equiv in results:
            f.write(f"{lbl}\t{truth}\t{pred}\t{equiv}\n")
    # calculate the ratio of semantic equivalence to true
    total = len(results)
    correct = sum(1 for item in results if item[3].lower() == 'true')
    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"Semantic equivalence accuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__=='__main__':
    main()