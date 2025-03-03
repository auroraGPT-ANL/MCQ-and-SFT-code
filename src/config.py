#!/usr/bin/env python

import os
import yaml
import logging

# Set up a unique logger.
logger = logging.getLogger("MCQGenerator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# run this from top level of repo
def load_config(file_path="config.yml"):
    """
    Safely load configuration settings from a YAML file.
    """
    if not os.path.exists(file_path):
        logger.error(f"Config file '{file_path}' not found.")
        raise FileNotFoundError(f"Config file '{file_path}' not found.")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file '{file_path}': {exc}")
        raise

# Load the configuration.
_config = load_config()

# Group variables logically.
prompts =      _config.get("prompts", {})
model_config = _config.get("model", {})
quality =      _config.get("quality", {})

# Prompt settings.
system_message     = prompts.get("system_message", "")
user_message       = prompts.get("user_message", "")
system_message_2   = prompts.get("system_message_2", "")
user_message_2     = prompts.get("user_message_2", "")
system_message_3   = prompts.get("system_message_3", "")
user_message_3     = prompts.get("user_message_3", "")

# Model config
defaultModel = model_config.get("name", "alcf:mistralai/Mistral-7B-Instruct-v0.3")
defaultTemperature = model_config.get("temperature", 0.7)
defaultBaseModel =   model_config.get("baseModel", "None")
defaultTokenizer =   model_config.get("Tokenizer", "None")

# Quality (MCQ) settings
minScore           = quality.get("minScore", 7)

# Directories for user data
directories = _config.get("directories", {})
papers_dir  = directories.get("papers", "_PAPERS")
json_dir    = directories.get("json", "_JSON")
mcq_dir     = directories.get("mcq", "_MCQ")
results_dir = directories.get("results", "_RESULTS")

