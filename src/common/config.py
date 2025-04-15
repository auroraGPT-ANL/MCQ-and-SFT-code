#!/usr/bin/env python

import os
import yaml
import logging
import threading
import signal

# Global lock for file operations.
output_file_lock = threading.Lock()

# Set up a unique logger.
logger = logging.getLogger("MCQGenerator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Global flag for graceful shutdown.
shutdown_event = threading.Event()

def handle_sigint(signum, frame):
    shutdown_event.set()
    logger.warning("Interrupt or fatal error: exiting after all threads complete.")

signal.signal(signal.SIGINT, handle_sigint)

def initiate_shutdown(message="Shutting down."):
    logger.error(message)
    shutdown_event.set()
    raise SystemExit(message)

# "No-op" progress bar for quiet mode.
class NoOpTqdm:
    """A do-nothing progress bar class that safely ignores all tqdm calls."""
    def __init__(self, total=0, desc="", unit=""):
        self.total = total
        self.n = 0

    def update(self, n=1):
        self.n += n

    def set_postfix_str(self, s):
        pass

    def close(self):
        pass

def configure_verbosity(args):
    """
    Set logging level and decide whether to use a progress bar based on command-line arguments.
    Returns:
        use_progress_bar (bool): True if a progress bar should be used, False otherwise.
    """
    if args.verbose:
        logger.setLevel(logging.INFO)
        use_progress_bar = False
        logger.info("Verbose mode enabled.")
    elif args.quiet:
        logger.setLevel(logging.CRITICAL)
        use_progress_bar = False
        logger.info("Quiet mode enabled.")
    else:
        logger.setLevel(logging.WARNING)
        use_progress_bar = True
        logger.info("Default progress bar mode enabled.")
    return use_progress_bar

def load_config(file_path=None):
    """
    Safely load configuration settings from a YAML file.
    If file_path is not provided, compute the path relative to the repository root.
    """
    # Compute the directory of this file.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # Our structure:
    # repo_root/
    #   config.yml
    #   src/
    #     common/config.py  <--- this file
    # So repo_root is two levels above this file.
    repo_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    if file_path is None:
        file_path = os.path.join(repo_root, "config.yml")

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

def load_secrets(config_data):
    """
    Load secrets from the file specified in config.yml's argo.username_file.
    Returns the username or None if not found.
    """
    argo_config = config_data.get("argo", {})
    secrets_file = argo_config.get("username_file")

    if not secrets_file:
        logger.warning("No secrets file specified in config.yml.")
        return None

    try:
        with open(secrets_file, "r", encoding="utf-8") as f:
            secrets = yaml.safe_load(f)
            return secrets.get("argo", {}).get("username")
    except FileNotFoundError:
        logger.warning(f"Secrets file '{secrets_file}' not found.")
        return None
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing secrets file '{secrets_file}': {exc}")
        return None
    except Exception as e:
        logger.error(f"Error reading secrets file: {e}")
        return None

# Load the raw YAML data from config.yml at the repository root.
_config = load_config()

# Load Argo username from secrets.
argo_user = load_secrets(_config)
if not argo_user:
    logger.warning("Argo username not found in secrets file.")

# --- Model dictionaries ---
model   = _config.get("model", {})
model_b = _config.get("model_b", {})
model_c = _config.get("model_c", {})
model_d = _config.get("model_d", {})

defaultModel  = model.get("name", "alcf:meta-llama/Meta-Llama-3-70B-Instruct")
defaultModelB = model_b.get("name", "alcf:mistralai/Mistral-7B-Instruct-v0.3")

# --- Standard MCQ generation prompts ---
prompts = _config.get("prompts", {})
system_message   = prompts.get("system_message", "")
user_message     = prompts.get("user_message", "")
system_message_2 = prompts.get("system_message_2", "")
user_message_2   = prompts.get("user_message_2", "")
system_message_3 = prompts.get("system_message_3", "")
user_message_3   = prompts.get("user_message_3", "")

# --- Fact extraction prompts ---
fact_extraction_system = prompts.get("fact_extraction_system", "")
fact_extraction_user   = prompts.get("fact_extraction_user", "")

# --- Scoring prompts for score_answers.py ---
scoring_prompts       = _config.get("scoring_prompts", {})
score_main_system     = scoring_prompts.get("main_system", "")
score_main_prompt     = scoring_prompts.get("main_prompt", "")
score_fallback_system = scoring_prompts.get("fallback_system", "")
score_fallback_prompt = scoring_prompts.get("fallback_prompt", "")

# --- Prompts for generate_nugget.py ---
nugget_prompts = _config.get("nugget_prompts", {})

timeout       = _config.get("timeout", 60)
quality       = _config.get("quality", {})
minScore      = quality.get("minScore", 7)
chunkSize     = quality.get("chunkSize", 1024)
saveInterval  = quality.get("saveInterval", 50)
defaultThreads = quality.get("defaultThreads", 4)

# --- Data Directories ---
# Compute the repository root (same as in load_config)
this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
directories = _config.get("directories", {})

papers_dir  = os.path.join(repo_root, directories.get("papers", "_PAPERS"))
json_dir    = os.path.join(repo_root, directories.get("json", "_JSON"))
mcq_dir     = os.path.join(repo_root, directories.get("mcq", "_MCQ"))
results_dir = os.path.join(repo_root, directories.get("results", "_RESULTS"))

