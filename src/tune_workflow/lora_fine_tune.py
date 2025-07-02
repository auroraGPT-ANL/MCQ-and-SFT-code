#!/usr/bin/env python

import argparse
import os
import torch
from trl import SFTTrainer
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
import json
from huggingface_hub import login


def main():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on a given JSON dataset, then merge and save.")
    parser.add_argument('-d', "--dataset_file", type=str, required=True, help="Path to the JSON file containing the dataset (e.g. text.json).")
    parser.add_argument('-o', "--output_dir", type=str, required=True, help="Output directory for saving the final model.")
    args = parser.parse_args()

    dataset_file = args.dataset_file
    output_dir = args.output_dir

    # -------------------------------------------------------------------------
    # 1. Log in to Hugging Face
    # -------------------------------------------------------------------------
    try:
        with open("hf_access_token.txt", "r") as file:
            hf_access_token = file.read().strip()
        login(hf_access_token)
    except FileNotFoundError:
        print("ERROR: 'hf_access_token.txt' not found. Please create this file with a valid Hugging Face access token.")
        return

    model_name = "meta-llama/Llama-3.1-8B-Instruct"


    max_seq_length = 2048

    # -------------------------------------------------------------------------
    # 2. Load dataset
    # -------------------------------------------------------------------------
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    num_rows = dataset.num_rows
    print(f"Number of rows: {num_rows}")
    num_steps = num_rows % 4

    # -------------------------------------------------------------------------
    # 3. Load model and tokenizer with quantization
    # -------------------------------------------------------------------------
    try:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config,
            device_map="auto",
        )
    except Exception as e:
        print(f"ERROR: Failed to download or load the model '{model_name}'.")
        print(f"Details: {e}")
        return

    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    base_model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    base_model = prepare_model_for_kbit_training(base_model)

    # -------------------------------------------------------------------------
    # 4. Create and wrap PEFT LoRA model
    # -------------------------------------------------------------------------
    lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=32,
        lora_dropout=0.05
    )
    peft_model = get_peft_model(base_model, lora_config)

    # -------------------------------------------------------------------------
    # 5. Train with SFTTrainer
    # -------------------------------------------------------------------------
    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=num_steps,
            bf16=True,
            logging_steps=1,
            output_dir="SFT-outputs",
            optim="adamw_8bit",
            seed=3407,
        ),
    )

    trainer.train()

    # -------------------------------------------------------------------------
    # 6. Save adapter and tokenizer
    # -------------------------------------------------------------------------
    peft_model.save_pretrained(output_dir, save_adapter=True, save_config=True)
    tokenizer.save_pretrained(output_dir)

    # -------------------------------------------------------------------------
    # 7. Merge LoRA weights with base model
    # -------------------------------------------------------------------------
    try:
        model_to_merge = peft_model.from_pretrained(
            AutoModelForCausalLM.from_pretrained(model_name).to("cuda"),
            output_dir
        )
        merged_model = model_to_merge.merge_and_unload()
        merged_model.save_pretrained(output_dir, save_method="merged_16bit")
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print("WARNING: Failed to merge LoRA weights with base model.")
        print(f"Details: {e}")

    # -------------------------------------------------------------------------
    # 8. Push to Hugging Face Hub (optional)
    # -------------------------------------------------------------------------
    try:
        repo_id = "ianfoster/" + os.path.basename(output_dir)
        merged_model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
    except Exception as e:
        print("NOTE: Skipping Hugging Face push (optional).")
        print(f"Details: {e}")


if __name__ == "__main__":
    main()

