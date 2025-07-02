#!/usr/bin/env python
"""
Create user-specific *config.local.yml* and *secrets.yml* skeletons.

This script now **backs up** existing files when running with `--force`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import textwrap
import yaml

# Paths for user-specific config files
CONFIG_LOCAL = Path("config.local.yml")
SECRETS_YML = Path("secrets.yml")

# Sample templates for config.local.yml and secrets.yml
SAMPLE_CONFIG = textwrap.dedent(
    """\
# ▶️  Models for this run
workflow:
  extraction: openai:gpt-4o
  contestants: [openai:gpt-4o, argo:mcq-8b]
  target: openai:gpt-4o
"""
)

SAMPLE_SECRETS = textwrap.dedent(
    """\
# ▶️  Credentials (never commit this!)
# openai_api_key: "sk-..."
# argo_username:  "user"
# argo_token:     "token"
"""
)

def write_file(path: Path, content: str, force: bool) -> None:
    """
    Write `content` to `path`.

    If `force` is True and the file exists, back it up to `<path>.bak` before overwriting.
    If `force` is False and the file exists, skip writing.
    """
    if path.exists():
        if force:
            backup_path = path.with_suffix(path.suffix + ".bak")
            try:
                shutil.copy(path, backup_path)
                print(f"⚠️  Backed up {path} to {backup_path}")
            except Exception as e:
                print(f"⚠️  Could not back up {path}: {e}")
        else:
            print(f"⏭  {path} already exists — skipped")
            return
    path.write_text(content, encoding="utf-8")
    print(f"✔  Wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Init local config skeletons (with backup)"
    )
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="overwrite and back up existing files"
    )
    args = parser.parse_args(argv)

    write_file(CONFIG_LOCAL, SAMPLE_CONFIG, args.force)
    write_file(SECRETS_YML, SAMPLE_SECRETS, args.force)


if __name__ == "__main__":
    main()

