#!/usr/bin/env python

"""common/config.py – central configuration loader

* Loads `config.yml` from the repository root.
* Loads **all** secrets from `secrets.yml` (path given in `config.yml`).
* Exposes `get_secret("dot.path")` so callers can fetch any secret without
  changing this file.
"""

from __future__ import annotations

import os
import sys
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
CONFIG_LOCAL_YML_PATH = os.path.join(REPO_ROOT, "config.local.yml")

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
output_file_lock = threading.Lock()
shutdown_event = threading.Event()


def _handle_sigint(signum, frame):
    shutdown_event.set()
    logger.warning("Interrupt received – shutting down after workers finish (could take 60-90s)")


def initiate_shutdown(message: str = "Shutting down.") -> None:
    logger.error(message)
    shutdown_event.set()
    sys.exit(1)


signal.signal(signal.SIGINT, _handle_sigint)

# ---------------------------------------------------------------------------
#  Progress‑bar stub
# ---------------------------------------------------------------------------
class NoOpTqdm:
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

def configure_verbosity(args) -> bool:
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
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        logger.error(f"\n❌ YAML parsing error in '{path}':\n{exc}")
        initiate_shutdown(f"YAML parsing failed for '{path}' – please check indentation and syntax.")


def load_config(path: str | None = None) -> dict[str, Any]:
    path = path or CONFIG_YML_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"config '{path}' not found")
    return _safe_load_yaml(path)


def load_local_config() -> dict[str, Any]:
    if not os.path.exists(CONFIG_LOCAL_YML_PATH):
        return {}
    return _safe_load_yaml(CONFIG_LOCAL_YML_PATH)


def load_secrets(cfg: dict[str, Any]) -> dict[str, Any]:
    rel_path = cfg.get("argo", {}).get("username_file")
    if not rel_path:
        logger.warning("No secrets file specified in config.yml")
        return {}
    sp = rel_path if os.path.isabs(rel_path) else os.path.join(REPO_ROOT, rel_path)
    if not os.path.exists(sp):
        logger.warning(f"Secrets file '{sp}' not found")
        return {}
    return _safe_load_yaml(sp)

# ---------------------------------------------------------------------------
#  Public helper to fetch secrets
# ---------------------------------------------------------------------------

def get_secret(path: str, default: Any = None) -> Any:
    node = _SECRETS
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node

# ---------------------------------------------------------------------------
#  Load config + secrets once
# ---------------------------------------------------------------------------

_CONFIG = load_config()
_LOCAL_CONFIG = load_local_config()
_SECRETS = load_secrets(_CONFIG)

try:
    _SERVERS = load_config("servers.yml")
    cels_model_servers = _SERVERS.get("servers", {})
except FileNotFoundError:
    cels_model_servers = None

argo_user = get_secret("argo.username")
openai_access_token = get_secret("openai.access_token")

# ---------------------------------------------------------------------------
#  Workflow configuration
# ---------------------------------------------------------------------------
_workflow_cfg = _CONFIG.get("workflow", {}).copy()
_workflow_cfg.update(_LOCAL_CONFIG.get("workflow", {}))
workflow = _workflow_cfg

# ---------------------------------------------------------------------------
#  Unpack frequently‑used config fields
# ---------------------------------------------------------------------------

model_type_endpoints = _CONFIG.get("model_type_endpoints", {})

http_client = _CONFIG.get("http_client", {
    "connect_timeout": 3.05,
    "read_timeout": 10,
    "max_retries": 1,
    "pool_connections": 1,
    "pool_maxsize": 1
})

model   = _LOCAL_CONFIG.get("model", _CONFIG.get("model", {}))
model_b = _LOCAL_CONFIG.get("model_b", _CONFIG.get("model_b", {}))
model_c = _LOCAL_CONFIG.get("model_c", _CONFIG.get("model_c", {}))
model_d = _LOCAL_CONFIG.get("model_d", _CONFIG.get("model_d", {}))

def _model_name(d):
    return d.get("name") if isinstance(d, dict) else None

defaultModel  = _model_name(model)   or _CONFIG.get("defaultModel")
defaultModelB = _model_name(model_b) or _CONFIG.get("defaultModelB")
defaultModelC = _model_name(model_c)
defaultModelD = _model_name(model_d)

_dirs = _CONFIG.get("directories", {})
papers_dir  = os.path.join(REPO_ROOT, _dirs.get("papers",   "_PAPERS"))
json_dir    = os.path.join(REPO_ROOT, _dirs.get("json",     "_JSON"))
mcq_dir     = os.path.join(REPO_ROOT, _dirs.get("mcq",      "_MCQ"))
results_dir = os.path.join(REPO_ROOT, _dirs.get("results",  "_RESULTS"))

