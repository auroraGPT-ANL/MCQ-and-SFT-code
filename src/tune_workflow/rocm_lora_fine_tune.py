#!/usr/bin/env python

# rocm_lora_fine_tune - runs on AMD/ROCm GPU systems like Lumi

import argparse
import os
import torch
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
    parser.add_argument(
        '--rope-scaling-type', type=str, default="dynamic",
        help="RoPE scaling type (e.g., 'dynamic' or 'linear')."
    )
    parser.add_argument(
        '--rope-scaling-factor', type=float, default=8.0,
        help="RoPE scaling factor (e.g., 8.0)."
    )
    args = parser.parse_args()

    dataset_file = args.dataset_file
    output_dir = args.output_dir
    model_name = args.model_name

    # -------------------------------------------------------------------------
    # 1. Log in to Hugging Face
    # -------------------------------------------------------------------------
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("ERROR: HUGGINGFACE_TOKEN environment variable is not set.")
        return
    login(hf_token)

    # -------------------------------------------------------------------------
    # 2. Configure distributed training if needed
    # -------------------------------------------------------------------------
    if "RANK" not in os.environ or "MASTER_ADDR" not in os.environ:
        print("⚠️  No distributed training environment detected — falling back to single-node mode.")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

    # -------------------------------------------------------------------------
    # 3. Load dataset
    # -------------------------------------------------------------------------
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    num_rows = dataset.num_rows
    print(f"Number of rows: {num_rows}")
    num_steps = max(1, num_rows // 4)

    # -------------------------------------------------------------------------
    # 4. Load model and tokenizer with explicit rope_scaling override
    # -------------------------------------------------------------------------
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        config = AutoConfig.from_pretrained(model_name, token=hf_token)

        # Prepare minimal valid rope_scaling dict
        rope_kwargs = {
            "type": args.rope_scaling_type,
            "factor": args.rope_scaling_factor
        }

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            device_map="auto",
            token=hf_token,
            torch_dtype=torch.float16,
            rope_scaling=rope_kwargs
        )
    except Exception as e:
        print(f"ERROR: Failed to download or load the model '{model_name}'.")
        print(f"Details: {e}")
        return

    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    base_model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    # -------------------------------------------------------------------------
    # 5. Create and wrap PEFT LoRA model
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
    # 6. Train with SFTTrainer
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
            optim="adamw_torch",
            seed=3407,
        ),
    )

    trainer.train()

    # -------------------------------------------------------------------------
    # 7. Save adapter and tokenizer
    # -------------------------------------------------------------------------
    peft_model.save_pretrained(output_dir, save_adapter=True, save_config=True)
    tokenizer.save_pretrained(output_dir)

    # -------------------------------------------------------------------------
    # 8. Merge LoRA weights with base model
    # -------------------------------------------------------------------------
    try:
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(output_dir, save_method="merged_16bit")
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print("WARNING: Failed to merge LoRA weights with base model.")
        print(f"Details: {e}")

    # -------------------------------------------------------------------------
    # 9. Push to Hugging Face Hub (optional)
    # -------------------------------------------------------------------------
    try:
        repo_id = "ianfoster/" + os.path.basename(output_dir)
        merged.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
    except Exception as e:
        print("NOTE: Skipping Hugging Face push (optional).")
        print(f"Details: {e}")


if __name__ == "__main__":
    main()

