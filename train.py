"""
QLoRA Fine-tuning of Llama-3.1-8B on Magicoder-Evol-Instruct-110K
for code generation. Tracks experiments with W&B, pushes model to HuggingFace Hub.
"""

import os
import yaml
import argparse
import wandb
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_bnb_config(qlora_config: dict) -> BitsAndBytesConfig:
    """Create BitsAndBytes config for 4-bit quantisation (QLoRA)."""
    return BitsAndBytesConfig(
        load_in_4bit=qlora_config["load_in_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(qlora_config: dict) -> LoraConfig:
    """Create LoRA config."""
    return LoraConfig(
        r=qlora_config["r"],
        lora_alpha=qlora_config["lora_alpha"],
        lora_dropout=qlora_config["lora_dropout"],
        target_modules=qlora_config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(model_config: dict, bnb_config: BitsAndBytesConfig):
    """Load base model in 4-bit and tokenizer."""
    print(f"Loading model: {model_config['base_model']}")
    model_name = model_config["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model, tokenizer


class CodeGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, num_samples=3):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        model.eval()
        table = wandb.Table(columns=["step", "instruction", "real code", "generated_code"])

        for i in range(self.num_samples):
            text = self.eval_dataset[i]["text"]

            instruction = text.split("### Response:")[0] + "### Response:"
            real_code = text.split("### Response:")[1]

            inputs = self.tokenizer(instruction, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                        input_ids=inputs["input_ids"].to(model.device),
                        attention_mask=inputs["attention_mask"].to(model.device),
                        max_new_tokens=200,    # generate up to 200 new tokens
                        temperature=0.7,       # low temperature = more focused output
                        do_sample=True,        # use sampling not greedy
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            table.add_data(state.global_step, instruction, real_code, generated_text)

        wandb.log({"code_samples": table})
        model.train()


def main(config_path: str):
    # Load config
    config = load_config(config_path)
    model_config = config["model"]
    qlora_config = config["qlora"]
    training_config = config["training"]
    dataset_config = config["dataset"]
    wandb_config = config["wandb"]

    # Initialise W&B
    wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        config=config,
        name=f"llama-3-1-8b-qlora-magicoder-{training_config['epochs']}ep",
    )

    # Load processed dataset
    print("Loading processed dataset...")
    train_dataset = load_from_disk("./data/processed/train")
    eval_dataset = load_from_disk("./data/processed/test")
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # Set up QLoRA
    bnb_config = get_bnb_config(qlora_config)
    lora_config = get_lora_config(qlora_config)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config, bnb_config)

    # Initialise W&B callback for logging code samples
    callback = CodeGenerationCallback(
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        num_samples=3
    )


    # Training arguments
    training_args = SFTConfig(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        warmup_steps=training_config["warmup_steps"],
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        report_to="wandb",
        push_to_hub=training_config["push_to_hub"],
        hub_model_id=training_config["hub_model_id"],
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        max_length=training_config["max_seq_length"],
        dataset_text_field="text",
        loss_type="nll", # disables chunked CE which causes the bug
    )

    # Initialise trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[callback],
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print("Saving model...")
    trainer.save_model(training_config["output_dir"])

    # Push to HuggingFace Hub
    if training_config["push_to_hub"]:
        print("Pushing to HuggingFace Hub...")
        trainer.push_to_hub()

    wandb.finish()
    print("Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qlora_config.yaml", help="Path to config file")
    args = parser.parse_args()

    main(args.config)