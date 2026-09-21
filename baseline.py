"""
Baseline evaluation of Llama-3.1-8B (zero-shot) on HumanEval+ and MBPP+
Loads model in 4-bit, generates solutions, scores with evalplus.
"""

import json
import os
import argparse
import torch
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from train import load_config, get_bnb_config


def get_problems(dataset_name: str):
    """Load problems for the given dataset."""
    if dataset_name == "humaneval":
        return get_human_eval_plus()
    elif dataset_name == "mbpp":
        return get_mbpp_plus()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_baseline_on_dataset(model, tokenizer, dataset_name: str, max_new_tokens: int):
    """Run baseline evaluation on a single dataset."""
    problems = get_problems(dataset_name)
    print(f"\nEvaluating {len(problems)} {dataset_name} problems...")

    solutions = []
    for i, (task_id, problem) in enumerate(problems.items()):
        prompt = problem["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        solution = tokenizer.decode(new_tokens, skip_special_tokens=True)

        solutions.append({
            "task_id": task_id,
            "solution": prompt + solution
        })
        print(f"[{i+1}/{len(problems)}] {task_id} done")

    os.makedirs("results", exist_ok=True)
    solutions_file = f"results/baseline_{dataset_name}_solutions.jsonl"
    with open(solutions_file, "w") as f:
        for sol in solutions:
            f.write(json.dumps(sol) + "\n")

    print(f"Solutions saved! Running evalplus scoring on {dataset_name}...")

    result = subprocess.run([
        "python", "-m", "evalplus.evaluate",
        "--dataset", dataset_name,
        "--samples", solutions_file
    ], capture_output=True, text=True)

    print(result.stdout)

    with open(f"results/baseline_{dataset_name}_results.json", "w") as f:
        json.dump({"stdout": result.stdout}, f, indent=2)

    return result.stdout


def run_baseline(config_path: str, dataset_override: str = None):
    config = load_config(config_path)
    model_name = config["model"]["base_model"]
    qlora_config = config["qlora"]
    eval_config = config["evaluation"]

    datasets = [dataset_override] if dataset_override else eval_config["datasets"]
    max_new_tokens = eval_config["max_new_tokens"]

    print(f"Loading {model_name} in 4-bit...")
    bnb_config = get_bnb_config(qlora_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    print("Model loaded!")

    # evaluate on all datasets
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")
        run_baseline_on_dataset(model, tokenizer, dataset_name, max_new_tokens)

    print("\nAll baseline evaluations complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qlora_config.yaml")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset to evaluate on. If not specified, runs all datasets from config.")
    args = parser.parse_args()
    run_baseline(args.config, args.dataset)