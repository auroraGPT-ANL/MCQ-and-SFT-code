#!/usr/bin/env python

"""common/config.py – central configuration loader

* Loads `config.yml` from the repository root.
* Loads **all** secrets from `secrets.yml` (path given in `config.yml`).
* Exposes `get_secret("dot.path")` so callers can fetch any secret without
  changing this file.
"""

from __future__ import annotations

import os
import signal
import threading
import logging
import yaml
from typing import Any

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # src/common
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))
CONFIG_YML_PATH = os.path.join(REPO_ROOT, "config.yml")

# ---------------------------------------------------------------------------
#  Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("MCQGenerator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
#  Graceful‑shutdown helpers
# ---------------------------------------------------------------------------
output_file_lock = threading.Lock()  # global lock for file I/O
shutdown_event = threading.Event()


def _handle_sigint(signum, frame):  # noqa: D401, unused‑arg
    """Set the shutdown flag so worker threads can exit cleanly."""
    shutdown_event.set()
    logger.warning("Interrupt received – shutting down after workers finish (could take 60-90s)")


def initiate_shutdown(message: str = "Shutting down.") -> None:
    logger.error(message)
    shutdown_event.set()
    raise SystemExit(message)


signal.signal(signal.SIGINT, _handle_sigint)

# ---------------------------------------------------------------------------
#  Progress‑bar stub for quiet mode
# ---------------------------------------------------------------------------
class NoOpTqdm:  # noqa: D101
    def __init__(self, total: int = 0, desc: str = "", unit: str = ""):
        self.total = total
        self.n = 0

    def update(self, n: int = 1):
        self.n += n

    def set_postfix_str(self, _s: str):
        pass

    def close(self):
        pass

# ---------------------------------------------------------------------------
#  Verbosity helper
# ---------------------------------------------------------------------------

def configure_verbosity(args) -> bool:  # noqa: ANN001 – argparse.Namespace
    """Return *True* if a progress‑bar should be used based on CLI flags."""
    if getattr(args, "verbose", False):
        logger.setLevel(logging.INFO)
        logger.info("Verbose mode.")
        return False
    if getattr(args, "quiet", False):
        logger.setLevel(logging.CRITICAL)
        logger.info("Quiet mode.")
        return False
    logger.setLevel(logging.WARNING)
    logger.info("Default progress‑bar mode.")
    return True

# ---------------------------------------------------------------------------
#  YAML loaders
# ---------------------------------------------------------------------------

def _safe_load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load *config.yml* (or a custom path)."""
    path = path or CONFIG_YML_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"config '{path}' not found")
    try:
        return _safe_load_yaml(path)
    except yaml.YAMLError as exc:  # pragma: no cover
        logger.error(f"Error parsing YAML file '{path}': {exc}")
        raise


def load_secrets(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load the secrets file referred to in *config.yml* (returns empty dict if absent)."""
    rel_path = cfg.get("argo", {}).get("username_file")
    if not rel_path:
        logger.warning("No secrets file specified in config.yml")
        return {}
    secrets_path = rel_path if os.path.isabs(rel_path) else os.path.join(REPO_ROOT, rel_path)
    if not os.path.exists(secrets_path):
        logger.warning(f"Secrets file '{secrets_path}' not found")
        return {}
    try:
        return _safe_load_yaml(secrets_path)
    except yaml.YAMLError as exc:
        logger.error(f"YAML error in secrets file '{secrets_path}': {exc}")
        return {}


# ---------------------------------------------------------------------------
#  Public helper to fetch secrets
# ---------------------------------------------------------------------------

def get_secret(path: str, default: Any = None) -> Any:
    """Retrieve a secret with dotted notation, e.g. ``get_secret('argo.username')``."""
    node: Any = _SECRETS
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node

# ---------------------------------------------------------------------------
#  Load config + secrets once at import time
# ---------------------------------------------------------------------------

_CONFIG = load_config()
_SECRETS = load_secrets(_CONFIG)

# convenience for legacy code
argo_user = get_secret("argo.username")
if not argo_user:
    logger.warning("Argo username not found in secrets file.")

# convenience for OpenAI
openai_access_token = get_secret("openai.access_token")
if not openai_access_token:
    logger.warning(
        "OpenAI access token not found in secrets file; "
        "will fall back to openai_access_token.txt"
    )


# ---------------------------------------------------------------------------
#  Unpack frequently‑used config fields
# ---------------------------------------------------------------------------

# Models
model   = _CONFIG.get("model", {})
model_b = _CONFIG.get("model_b", {})
model_c = _CONFIG.get("model_c", {})
model_d = _CONFIG.get("model_d", {})

def _model_name(d):
    return d.get("name") if isinstance(d, dict) else None

defaultModel  = _model_name(model)   or _CONFIG.get("defaultModel")
defaultModelB = _model_name(model_b) or _CONFIG.get("defaultModelB")
defaultModelC = _model_name(model_c)
defaultModelD = _model_name(model_d)

# Prompts
prompts = _CONFIG.get("prompts", {})
system_message   = prompts.get("system_message", "")
user_message     = prompts.get("user_message", "")
system_message_2 = prompts.get("system_message_2", "")
user_message_2   = prompts.get("user_message_2", "")
system_message_3 = prompts.get("system_message_3", "")
user_message_3   = prompts.get("user_message_3", "")

# Fact‑extraction prompts
fact_extraction_system = prompts.get("fact_extraction_system", "")
fact_extraction_user   = prompts.get("fact_extraction_user", "")

# Scoring prompts
scoring_prompts       = _CONFIG.get("scoring_prompts", {})
score_main_system     = scoring_prompts.get("main_system", "")
score_main_prompt     = scoring_prompts.get("main_prompt", "")
score_fallback_system = scoring_prompts.get("fallback_system", "")
score_fallback_prompt = scoring_prompts.get("fallback_prompt", "")

# Nugget prompts
nugget_prompts = _CONFIG.get("nugget_prompts", {})

# Runtime parameters
timeout        = _CONFIG.get("timeout", 60)
quality        = _CONFIG.get("quality", {})
minScore       = quality.get("minScore", 7)
chunkSize      = quality.get("chunkSize", 1024)
saveInterval   = quality.get("saveInterval", 50)
defaultThreads = quality.get("defaultThreads", 4)

# Data directories
_dirs = _CONFIG.get("directories", {})
papers_dir  = os.path.join(REPO_ROOT, _dirs.get("papers",   "_PAPERS"))
json_dir    = os.path.join(REPO_ROOT, _dirs.get("json",     "_JSON"))
mcq_dir     = os.path.join(REPO_ROOT, _dirs.get("mcq",      "_MCQ"))
results_dir = os.path.join(REPO_ROOT, _dirs.get("results",  "_RESULTS"))

