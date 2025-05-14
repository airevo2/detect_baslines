from ppl_and_ngram_utils_llava import *
import argparse
import os

gsm8k_dataset_names = [
    "GSM8K_rewritten-test-1",
    "GSM8K_rewritten-test-2",
    "GSM8K_rewritten-test-3",
    "GSM8K_rewritten-train-1",
    "GSM8K_rewritten-train-2",
    "GSM8K_rewritten-train-3",
    "orgn-GSM8K-test",
    "orgn-GSM8K-train",
    ]
math_dataset_names = [
    "MATH_rewritten-test-1",
    "MATH_rewritten-test-2",
    "MATH_rewritten-test-3",
    "MATH_rewritten-train-1",
    "MATH_rewritten-train-2",
    "MATH_rewritten-train-3",
    "orgn-MATH-train",
    "orgn-MATH-test",
]



if __name__ == "__main__":

    parser = argparse.ArgumentParser('Benchmark Leakage Detection based on PPL', add_help=False)
    parser.add_argument('--dataset_name', type=str, required=True, help='path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='path to model')
    parser.add_argument('--model_name', type=str, required=True, help='model name')
    parser.add_argument('--model_type', type=str, default = "base", help='model type: base or chat or multimodal')
    parser.add_argument('--device', type=str, required=True, help='device')
    parser.add_argument('--n', type=int, required=True, help='n-gram', default=5)
    parser.add_argument('--num_samples', type=int, default=0, help='Number of samples to use for testing; 0 means all data')
    args = parser.parse_args()


    if args.model_type == "multimodal":
        # Load LLaVA multimodal model and processor
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, llava_context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base=None,
            model_name=model_name
        )
        model = model.half().to(args.device)
        model.eval()
    else:
        model, tokenizer = load_model(args.model_path, args.device, args.model_type)

    if args.dataset_name == "gsm8k":
        dataset_names = gsm8k_dataset_names
    elif args.dataset_name == "math":
        dataset_names = math_dataset_names
    elif args.dataset_name == "mmlu":
        dataset_names = ["mmlu"]
    elif args.dataset_name == "mmlu_clean":
        dataset_names = ["mmlu_clean"]
    elif args.dataset_name == "arc":
        dataset_names = ["arc"]
    elif args.dataset_name == "mathqa":
        dataset_names = ["mathqa"]  
    elif args.dataset_name.endswith('.tsv'):
        dataset_names = [args.dataset_name]
    else:
        raise ValueError("Invalid dataset")


    k = 5  # num of starting point.
    results_ngrm_summary = {}

    for dataset_name in dataset_names:
        # 默认 safe_name 为完整名称或路径，后续可覆盖
        safe_name = dataset_name
        if dataset_name.endswith('.tsv'):
            # 自定义 TSV 数据集：路径直接为 dataset_path，且 safe_name 为文件名去除扩展
            dataset_path = dataset_name
            safe_name = os.path.splitext(os.path.basename(dataset_name))[0]
        elif "rewritten" in dataset_name:
            dataset_path = f'./data/rewritten/{dataset_name}.jsonl'
        elif "orgn" in dataset_name:
            dataset_path = f'./data/original/{dataset_name}.jsonl'
        elif "mmlu" == dataset_name:
            dataset_path = "mmlu"
        elif "mmlu_clean" == dataset_name:
            dataset_path = "mmlu_clean"
        elif "arc" == dataset_name:
            dataset_path = "arc"
        elif "mathqa" == dataset_name:
            dataset_path = "mathqa"
        else:
            # 其他情形保留原名
            dataset_path = dataset_name

        # 根据 num_samples 决定加载方式：>0 抽样，否则加载全部
        if args.num_samples > 0:
            dataset = load_data_from_jsonl(
                dataset_path,
                num_samples=args.num_samples,
                ngram=True,
                model_type=args.model_type
            )
        else:
            dataset = load_data_from_jsonl(
                dataset_path,
                ngram=True,
                model_type=args.model_type
            )
        
        # 构建输出文件名，使用 safe_name，提前创建目录
        output_file_ngram = f'./outputs/ngram/{args.n}gram-{args.model_name}-{safe_name}.jsonl'
        os.makedirs(os.path.dirname(output_file_ngram), exist_ok=True)
        if args.model_type == "multimodal":
            ngram_results = calculate_n_gram_accuracy(
                args.n, k, dataset, model, tokenizer, args.device,
                output_file_ngram, args.model_type, image_processor
            )
        else:
            ngram_results = calculate_n_gram_accuracy(
                args.n, k, dataset, model, tokenizer, args.device,
                output_file_ngram, args.model_type
            )
        print(f"{dataset_name} {args.n}_gram_accuracy: ", ngram_results["mean_n_grams"])
        results_ngrm_summary[f'{dataset_name}'] = ngram_results["mean_n_grams"]
        
    print(f"ngram acc of {args.model_name}")
    for key, value in results_ngrm_summary.items():
        print(f"{key}: {value}")