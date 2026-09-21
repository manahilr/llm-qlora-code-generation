"""
Prepare Magicoder-Evol-Instruct-110K dataset for QLoRA fine-tuning.
Downloads dataset, takes subset, formats for Llama, saves train/test splits.
"""
import argparse

import yaml
from datasets import load_dataset
import json
import os

def format_instruction(example):
    """Format each example into Llama instruction format."""
    return {
        "text": f"""### Instruction:
        {example['instruction']}

        ### Response:
        {example['response']}"""
    }

def prepare_data(
    dataset_name: str = "ise-uiuc/Magicoder-Evol-Instruct-110K",
    subset_size: int = 20000,
    test_size: float = 0.1,
    output_dir: str = "./data/processed",
    seed: int = 42
):
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    print(f"Full dataset size: {len(dataset)}")
    print(f"Taking subset of {subset_size} examples...")

    # Take subset
    dataset = dataset.shuffle(seed=seed).select(range(subset_size))

    # Format into Llama instruction format
    print("Formatting dataset...")
    dataset = dataset.map(format_instruction)

    # Split into train and test
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = split["train"]
    test_dataset = split["test"]

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.save_to_disk(f"{output_dir}/train")
    test_dataset.save_to_disk(f"{output_dir}/test")

    # Save a few examples to inspect
    examples = [train_dataset[i]["text"] for i in range(3)]
    with open(f"{output_dir}/sample_examples.json", "w") as f:
        json.dump(examples, f, indent=2)

    print(f"\nDataset saved to {output_dir}")
    print("\nSample example:")
    print(examples[0][:500])
    print("...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qlora_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_config = config["dataset"]

    prepare_data(
        dataset_name=dataset_config["name"],
        subset_size=dataset_config["subset_size"],
        test_size=dataset_config["test_size"],
    )