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
    # -------------------------------------------------------------------------
    # Parse command-line arguments for dataset file and output directory
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser( description="Train a LoRA adapter on a given JSON dataset, then merge and save.")
    parser.add_argument( '-d', "--dataset_file", type=str, required=True, help="Path to the JSON file containing the dataset (e.g. text.json).")
    parser.add_argument( '-o', "--output_dir", type=str, required=True, help="Output directory for saving the final model.")
    parser.add_argument( '-v', "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument( '-m', '--model', default='meta-llama/Llama-3.1-8B-Instruct', help='Name of model to fine tune') # Base model from HF
    parser.add_argument( '-u', '--username', default='', help='User name to push to HF') 
    args = parser.parse_args()

    dataset_file = args.dataset_file
    output_dir   = args.output_dir
    verbose      = args.verbose
    model_name   = args.model
    username     = args.username

    # -------------------------------------------------------------------------
    # (Optional) Log in to Hugging Face
    # -------------------------------------------------------------------------
    # If you intend to push to HF Hub, ensure hf_access_token.txt is accessible
    with open("hf_access_token.txt", "r") as file:
        hf_access_token = file.read().strip()
    login(hf_access_token)

    max_seq_length = 2048  # e.g. for models that support RoPE scaling

    # -------------------------------------------------------------------------
    # Load the dataset
    # -------------------------------------------------------------------------
    if verbose: print(f'Step 1: Load dataset {dataset_file}')
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    num_rows = dataset.num_rows
    if verbose: print(f"Number of rows: {num_rows}")

    num_steps = num_rows % 4  # Just an example of how to define steps

    # -------------------------------------------------------------------------
    # Configure 4-bit quantization
    # -------------------------------------------------------------------------
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # -------------------------------------------------------------------------
    # Define the model and tokenizer
    # -------------------------------------------------------------------------
    if verbose: print(f'Step 2: Load model {model_name}')  
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config,
        device_map="auto",
    )

    # Fix for infinite generation issue (missing pad token)
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    base_model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    # Prepare for k-bit LoRA training
    base_model = prepare_model_for_kbit_training(base_model)

    # -------------------------------------------------------------------------
    # Create the PEFT LoRA configuration and wrap the base model
    # -------------------------------------------------------------------------
    if verbose: print(f'Step 3: Wrap base model')  
    lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=32,
        lora_dropout=0.05
    )

    peft_model = get_peft_model(base_model, lora_config)

    # -------------------------------------------------------------------------
    # Create and run the SFTTrainer
    # -------------------------------------------------------------------------
    if verbose: print(f'Step 4: Create and run SFTTrainer')  
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
            output_dir="SFT-outputs",  # You can change this if you want
            optim="adamw_8bit",
            seed=3407,
        ),
    )

    trainer.train()

    # -------------------------------------------------------------------------
    # Save the LoRA adapter + tokenizer to the user-specified output directory
    # -------------------------------------------------------------------------
    if verbose: print(f'Step 5: Save LoRA adapter and tokenizer to {output_dir}')  
    peft_model.save_pretrained(output_dir, save_adapter=True, save_config=True)
    tokenizer.save_pretrained(output_dir)

    # -------------------------------------------------------------------------
    # Merge LoRA weights with the base model and save the final merged model
    # -------------------------------------------------------------------------
    if verbose: print(f'Step 6: Merge LoRA weights with base model and save final merged model')  
    model_to_merge = peft_model.from_pretrained(
        AutoModelForCausalLM.from_pretrained(model_name).to("cuda"),
        output_dir
    )

    merged_model = model_to_merge.merge_and_unload()
    merged_model.save_pretrained(output_dir, save_method="merged_16bit")
    tokenizer.save_pretrained(output_dir)

    # -------------------------------------------------------------------------
    # Push merged model and tokenizer to Hugging Face Hub
    # -------------------------------------------------------------------------
    if username != '':
        if verbose: print(f'Step 7: Push merged model and tokenizer to Hugging Face Hub with username {username}')  
        merged_model.push_to_hub(f'{username}/{os.path.basename(output_dir)}')
        tokenizer.push_to_hub(f'{username}/{os.path.basename(output_dir)}')


if __name__ == "__main__":
    main()

