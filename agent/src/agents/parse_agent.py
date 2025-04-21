#!/usr/bin/env python
"""
ParseAgent

Agent that converts PDF files into JSON using the simple_parse utility.
"""
import os
from agents.agent_base import Agent
from common.simple_parse import process_directory
from common import config

class ParseAgent(Agent):
    """
    Agent for parsing PDFs into JSON files.
    """
    def run(self, context: dict) -> dict:
        # Source PDF directory (e.g. downloaded papers)
        pdf_dir = context.get("pdf_dir", config.papers_dir)
        # Target JSON output directory
        json_dir = context.get("parsed_dir", config.json_dir)
        os.makedirs(json_dir, exist_ok=True)

        # Determine whether to show a progress bar
        use_progress = context.get("v_flag", False)

        # Call the core parsing function directly
        process_directory(
            input_dir=pdf_dir,
            output_dir=json_dir,
            use_progress_bar=use_progress
        )

        # Return updated context key
        return {"parsed_dir": json_dir}

