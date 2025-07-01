#!/usr/bin/env python

import argparse
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
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using SFTTrainer on a JSON dataset.")
    parser.add_argument('-d', "--dataset_file", type=str, required=True, help="Path to the JSON file containing the dataset (e.g. text.json).")
    parser.add_argument('-o', "--output_name", type=str, required=True, help="Directory/name for the final model (e.g. llama-3.1-8B-merged-sft).")
    parser.add_argument('-u', "--user_name", type=str, help="Hugging Face user name for HF publication", default=None)

    return parser.parse_args()

def main():
    # -------------------------------------------------------------------------
    # 1. Parse command-line arguments
    # -------------------------------------------------------------------------
    args = parse_args()
    dataset_file = args.dataset_file
    output_name  = args.output_name
    user_name    = args.user_name

    print(f'Reading dataset {dataset_file} with results in {output_name}')

    # -------------------------------------------------------------------------
    # 2. (Optional) Log in to Hugging Face
    # -------------------------------------------------------------------------
    if user_name != None:
        with open('hf_access_token.txt', 'r') as file:
            hf_access_token = file.read().strip()
        login(hf_access_token)

    max_seq_length = 5120  # For example, supports RoPE scaling internally

    # Training configuration
    per_device_batch_size = 4
    gradient_accumulation_steps = 8

    # -------------------------------------------------------------------------
    # 3. Load the dataset from the user-specified file
    # -------------------------------------------------------------------------
    print(f"Loading dataset from {dataset_file}")
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    num_rows = dataset.num_rows
    print(f"Number of rows: {num_rows}")

    # Calculate steps based on effective batch size
    num_gpus = torch.cuda.device_count()
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps * num_gpus
    num_steps = num_rows // effective_batch_size
    print(f"Number of GPUs: {num_gpus}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Number of steps: {num_steps}")

    # -------------------------------------------------------------------------
    # 4. (Optional) Configure 4-bit quantization
    #    (Currently commented out; uncomment if needed)
    # -------------------------------------------------------------------------
    if 0: 
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        config = None

    # -------------------------------------------------------------------------
    # 5. Define the model and tokenizer
    # -------------------------------------------------------------------------
    #model_name = "meta-llama/Llama-3.1-8B-Instruct"
    #model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_name = "argonne-private/AuroraGPT-IT-v4-0125"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Quick check of tokenization lengths
    print("Analyzing token lengths...")
    sample = dataset.select(range(100))
    tokenized = tokenizer(sample['text'], truncation=False, return_length=True)
    lengths = [length for length in tokenized['length']]
    print(f"Max tokens: {max(lengths)}, 95th percentile: {np.percentile(lengths, 95):.0f}")

    # If you need quantization + device_map:
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     quantization_config=config,
    #     device_map="auto",
    # )
    # For now, just load normally:
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )

    # -------------------------------------------------------------------------
    # 6. Fix for infinite generation issue by adding/setting pad token
    # -------------------------------------------------------------------------
    print(f"Pad Token id: {tokenizer.pad_token_id} and Pad Token: {tokenizer.pad_token}")
    print(f"EOS Token id: {tokenizer.eos_token_id} and EOS Token: {tokenizer.eos_token}")

    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print(f"Pad Token id: {tokenizer.pad_token_id} and Pad Token: {tokenizer.pad_token}")
    print(f"EOS Token id: {tokenizer.eos_token_id} and EOS Token: {tokenizer.eos_token}")

    # -------------------------------------------------------------------------
    # 7. Create and run the SFTTrainer
    # -------------------------------------------------------------------------
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=10,
            max_steps=num_steps,
            bf16=True,
            logging_steps=1,
            output_dir="SFT-outputs",  # Intermediate checkpoints
            optim="adamw_8bit",
            seed=3407,
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
        ),
    )

    trainer.train(resume_from_checkpoint=False)


    # -------------------------------------------------------------------------
    # 8. Save final model and tokenizer
    # -------------------------------------------------------------------------
    base_model.save_pretrained(output_name)
    tokenizer.save_pretrained(output_name)

    # -------------------------------------------------------------------------
    # 9. (Optional) Push model and tokenizer to HF Hub
    # -------------------------------------------------------------------------
    if user_name != None:
        base_model.push_to_hub(f"{user_name}/{output_name}")
        tokenizer.push_to_hub(f"{user_name}/{output_name}")

if __name__ == "__main__":
    main()

