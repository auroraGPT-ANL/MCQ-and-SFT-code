#!/usr/bin/env python

# rocm_lora_fine_tune - runs on AMD/ROCm GPU systems like Lumi

import argparse
import os
import sys
from trl import SFTTrainer
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from huggingface_hub import login
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter on a given JSON dataset, then merge and save."
    )
    parser.add_argument(
        '-d', "--dataset_file", type=str, required=True,
        help="Path to the JSON file containing the dataset."
    )
    parser.add_argument(
        '-o', "--output_dir", type=str, required=True,
        help="Output directory for saving the final model."
    )
    parser.add_argument(
        '--model-name', type=str, default="meta-llama/Llama-3.1-8B",
        help="Model name to load from Hugging Face hub."
    )
    args = parser.parse_args()

    dataset_file = args.dataset_file
    output_dir = args.output_dir
    model_name = args.model_name

    # -------------------------------------------------------------------------
    # 0. Basic file checks
    # -------------------------------------------------------------------------
    if not os.path.isfile(dataset_file):
        print(f"ERROR: Dataset file not found: {dataset_file}")
        sys.exit(1)

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Could not create or access output directory '{output_dir}': {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 1. Log in to Hugging Face
    # -------------------------------------------------------------------------
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("ERROR: HUGGINGFACE_TOKEN environment variable is not set.")
        sys.exit(1)
    try:
        login(hf_token)
    except Exception as e:
        print(f"ERROR: Failed to login to Hugging Face Hub: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. Configure distributed training if needed
    # -------------------------------------------------------------------------
    if "RANK" not in os.environ or "MASTER_ADDR" not in os.environ:
        print("⚠️  No distributed training environment detected — falling back to single-node mode.")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")

    # -------------------------------------------------------------------------
    # 3. Load dataset
    # -------------------------------------------------------------------------
    try:
        dataset = load_dataset("json", data_files=dataset_file, split="train")
    except FileNotFoundError:
        print(f"ERROR: Could not find or open dataset file: {dataset_file}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        sys.exit(1)

    num_rows = getattr(dataset, "num_rows", None)
    if num_rows is None:
        print("WARNING: Could not determine number of rows in dataset.")
        num_steps = 1
    else:
        print(f"Number of rows: {num_rows}")
        num_steps = max(1, num_rows // 4)

    # -------------------------------------------------------------------------
    # 4. Load model and tokenizer with clean config (remove rope_scaling)
    # -------------------------------------------------------------------------
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        orig_config = AutoConfig.from_pretrained(model_name, use_auth_token=hf_token)
        config_dict = orig_config.to_dict()
        config_dict.pop("rope_scaling", None)
        clean_config = type(orig_config)(**config_dict)

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=clean_config,
            device_map="auto",
            use_auth_token=hf_token,
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"ERROR: Failed to download or load the model '{model_name}': {e}")
        sys.exit(1)

    # add padding token if missing
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
        base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    # -------------------------------------------------------------------------
    # 5. Create and wrap PEFT LoRA model
    # -------------------------------------------------------------------------
    try:
        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=32,
            lora_dropout=0.05
        )
        peft_model = get_peft_model(base_model, lora_config)
    except Exception as e:
        print(f"ERROR: Failed to initialize PEFT LoRA model: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 6. Train with SFTTrainer
    # -------------------------------------------------------------------------
    try:
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
                output_dir=os.path.join(output_dir, "SFT-outputs"),
                optim="adamw_torch",
                seed=3407,
            ),
        )
        trainer.train()
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 7. Save adapter and tokenizer
    # -------------------------------------------------------------------------
    try:
        peft_model.save_pretrained(output_dir, save_adapter=True, save_config=True)
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"ERROR: Failed to save adapter or tokenizer: {e}")

    # -------------------------------------------------------------------------
    # 8. Merge LoRA weights
    # -------------------------------------------------------------------------
    try:
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(output_dir, save_method="merged_16bit")
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print("WARNING: Failed to merge LoRA weights with base model.")
        print(f"Details: {e}")

    # -------------------------------------------------------------------------
    # 9. Push to Hugging Face Hub (optional)
    # -------------------------------------------------------------------------
    try:
        repo_id = "ianfoster/" + os.path.basename(output_dir.rstrip("/"))
        merged_model.push_to_hub(repo_id, use_auth_token=hf_token)
        tokenizer.push_to_hub(repo_id, use_auth_token=hf_token)
    except Exception as e:
        print("NOTE: Skipping Hugging Face push (optional).")
        print(f"Details: {e}")


if __name__ == "__main__":
    main()

