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

# new code for graceful exits
#
import signal
# Global flag for graceful shutdown
bail_out = False
def handle_sigint(signum, frame):
    global bail_out
    bail_out = True
    logger.warning("Interrupt received or fatal error encountered: setting bail_out flag to True.")
signal.signal(signal.SIGINT, handle_sigint)

def initiate_shutdown(message="Bailing gracefully"):
    logger.error(message)
    global bail_out
    bail_out = True
    raise SystemExit(message)


# add a "no op" progress bar for quiet mode
class NoOpTqdm:
    """A do-nothing progress bar class that safely ignores all tqdm calls."""
    def __init__(self, total=0, desc="", unit=""):
        self.total = total  # Store total count
        self.n = 0  # Keep track of progress count

    def update(self, n=1):
        self.n += n  # Simulate tqdm's progress tracking

    def set_postfix_str(self, s):
        pass  # No-op

    def close(self):
        pass  # No-op

##

def configure_verbosity(args):
    """
    Set logging level and decide whether to use a progress bar based on command-line arguments.
    Returns:
        use_progress_bar (bool): True if a progress bar should be used, False otherwise.
    """
    if args.verbose:
        logger.setLevel(logging.INFO)
        use_progress_bar = False
        logger.info("verbose mode")
    elif args.quiet:
        logger.setLevel(logging.CRITICAL)
        use_progress_bar = False
        logger.info("quiet mode")
    else:
        logger.setLevel(logging.WARNING)
        use_progress_bar = True
        logger.info("progress bar only mode")
    return use_progress_bar

##

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

# Load the raw YAML data

_config = load_config()

# --- Model dictionaries ---
model   = _config.get("model", {})
model_b = _config.get("model_b", {})
model_c = _config.get("model_c", {})
model_d = _config.get("model_d", {})

# In case the users does not define these in config.yml, we'll use these defaults
defaultModel = model.get("name", "alcf:meta-llama/Meta-Llama-3-70B-Instruct")
defaultModelB = model_b.get("name", "alcf:mistralai/Mistral-7B-Instruct-v0.3")

# --- Standard MCQ generation prompts ---
prompts = _config.get("prompts", {})
system_message        = prompts.get("system_message", "")
user_message          = prompts.get("user_message", "")
system_message_2      = prompts.get("system_message_2", "")
user_message_2        = prompts.get("user_message_2", "")
system_message_3      = prompts.get("system_message_3", "")
user_message_3        = prompts.get("user_message_3", "")

# --- Scoring prompts for score_answers.py ---
scoring_prompts       = _config.get("scoring_prompts", {})
score_main_system     = scoring_prompts.get("main_system", "")
score_main_prompt     = scoring_prompts.get("main_prompt", "")
score_fallback_system = scoring_prompts.get("fallback_system", "")
score_fallback_prompt = scoring_prompts.get("fallback_prompt", "")

# --- Other config values ---
quality               = _config.get("quality", {})
minScore              = quality.get("minScore", 7)
chunkSize             = quality.get("chunkSize", 1024)
saveInterval          = quality.get("saveInterval", 50) # for parallel_generate_answers.py
defaultThreads        = quality.get("defaultThreads", 4) 


directories           = _config.get("directories", {})
papers_dir            = directories.get("papers", "_PAPERS")
json_dir              = directories.get("json", "_JSON")
mcq_dir               = directories.get("mcq", "_MCQ")
results_dir           = directories.get("results", "_RESULTS")

