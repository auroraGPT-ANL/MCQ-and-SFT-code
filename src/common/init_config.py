#!/usr/bin/env python
"""
Create user-specific *config.local.yml* and *secrets.yml* skeletons.

Usage
-----
    python -m common.init_config
    python -m common.init_config --force      # overwrite if they exist
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap
import yaml


CONFIG_LOCAL = Path("config.local.yml")
SECRETS_YML = Path("secrets.yml")


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
    if path.exists() and not force:
        print(f"⏭  {path} already exists — skipped")
        return
    path.write_text(content, encoding="utf-8")
    print(f"✔  Wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Init local config skeletons")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="overwrite existing files",
    )
    args = parser.parse_args(argv)

    write_file(CONFIG_LOCAL, SAMPLE_CONFIG, args.force)
    write_file(SECRETS_YML, SAMPLE_SECRETS, args.force)


if __name__ == "__main__":
    main()

