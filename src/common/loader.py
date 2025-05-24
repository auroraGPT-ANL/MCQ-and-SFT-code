# src/common/loader.py
"""
File-system / YAML loader for Settings.

Usage
-----
    from common.loader import load_settings
    cfg = load_settings()

This is kept separate from the schema so that test
suites can import *models* without needing the repo's
YAML files on disk.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import ValidationError

from .schema import Settings  # the Pydantic model


# --------------------------------------------------------------------------- #
#  Constants – order matters
# --------------------------------------------------------------------------- #

REPO_CONFIG_FILES = [Path("config.yml"), Path("servers.yml")]
LOCAL_CONFIG_FILES = [Path("config.local.yml"), Path("secrets.yml")]


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Recursive dict merge – src wins."""
    for k, v in src.items():
        if k == "secrets":
            dst.setdefault(k, {})
            dst[k].update(v)
        elif isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:  # pragma: no cover
        print(f"⚠️  Error loading {path}: {e}")
        return {}


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #


def load_settings(extra_cfgs: Optional[List[Path]] = None) -> Settings:
    """
    Compose Settings from YAML + environment variables.

    Precedence (highest → lowest):
        1. Env vars  (AUGPT_WORKFLOW__EXTRACTION=…)
        2. Secrets section + secrets.yml
        3. config.local.yml
        4. servers.yml
        5. config.yml
    """
    data: Dict[str, Any] = {"secrets": {}}

    # repo-tracked defaults
    for p in REPO_CONFIG_FILES:
        _deep_merge(data, _safe_load_yaml(p))

    # user-specific overlays
    for p in LOCAL_CONFIG_FILES + (extra_cfgs or []):
        if p.name == "secrets.yml":
            data["secrets"].update(_safe_load_yaml(p))
        else:
            _deep_merge(data, _safe_load_yaml(p))

    # flatten nested secrets like argo.username → argo_username
    flattened = {}
    for top, blob in data.items():
        if top == "secrets":
            continue
        if isinstance(blob, dict):
            for subk, subv in blob.items():
                if subk in {"username", "access_token", "api_key", "token"}:
                    flattened[f"{top}_{subk}"] = subv
    data["secrets"].update(flattened)

    # env vars (dot-notation) – last override
    env_overlay: Dict[str, Any] = {}
    for key, value in os.environ.items():
        target = env_overlay
        if "." in key:
            *parents, leaf = key.split(".")
            for part in parents:
                target = target.setdefault(part, {})
            target[leaf] = value
        else:
            # non-dot keys are treated as secrets if not already set
            data["secrets"].setdefault(key, value)
    _deep_merge(data, env_overlay)

    # ---------------------------------------------------- validation
    try:
        return Settings.model_validate(data)
    except ValidationError as e:
        for err in e.errors():
            msg = err.get("msg", "")
            if "No endpoint configuration for shortname" in msg:
                missing = msg.split("'")[-2]
                print(
                    f"❌  Model '{missing}' not in servers.yml – "
                    "add an endpoint entry or pick another model."
                )
                sys.exit(1)
        raise  # re-throw for any other validation problem

