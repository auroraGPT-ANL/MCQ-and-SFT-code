# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains workflows for creating scientific training data from papers, specifically for Multiple Choice Question (MCQ) generation and New Knowledge Nugget (NKN) extraction. The codebase supports fine-tuning AI models using scientific papers as source material.

## Architecture

The codebase is organized into several key modules:

- **`src/common/`** - Shared utilities for model access, configuration management, PDF parsing, and authentication
- **`src/mcq_workflow/`** - MCQ generation pipeline components (generate, answer, score MCQs)
- **`src/nugget_workflow/`** - Knowledge nugget extraction and validation (under development)  
- **`src/tune_workflow/`** - Fine-tuning tools for models (under development)
- **`src/test/`** - Test infrastructure including stub models for offline testing
- **`agent/`** - Agent-based orchestration system (new architecture)
- **`legacy/`** - Original shell script-based workflow implementation

## Configuration System

The configuration uses a layered YAML approach with override precedence:

1. **`config.yml`** (tracked) - Stable defaults: prompts, directories, CLI defaults
2. **`servers.yml`** (tracked) - Shared endpoint catalogue for inference hosts  
3. **`config.local.yml`** (untracked) - Your runtime choices: models, per-user tweaks
4. **`secrets.yml`** (untracked) - Credentials: API keys, tokens referenced by servers.yml

Override precedence: `env vars` → `config.local.yml` → `servers.yml` → `config.yml`

## Environment Setup

**Conda Environment:**
```bash
conda env create -f environment.yml
conda activate augpt_env
```

**Python Path (required):**
```bash
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

**Working Directories:**
```bash
mkdir -p _PAPERS _JSON _MCQ _RESULTS
```

## Common Commands

### Main Workflow Execution

**Full MCQ workflow with 4 parallel workers:**
```bash
python -m mcq_workflow.run_workflow -p 4 -v
```

**Start from specific step:**
```bash
python -m mcq_workflow.run_workflow --step 3 -p 4
```

**Process subset of MCQs:**
```bash
python -m mcq_workflow.run_workflow -n 50 -p 4
```

### Individual Pipeline Components

**Convert PDFs to JSON:**
```bash
python -m common.simple_parse
```

**Generate MCQs:**
```bash
python -m mcq_workflow.generate_mcqs -p 4 -v
```

**Generate answers:**
```bash
python -m mcq_workflow.generate_answers -i MCQ-combined.json -m openai:gpt-4o -p 4
```

**Score answers:**
```bash
python -m mcq_workflow.score_answers -a openai:gpt-4o -b argo:mistral-7b -p 4
```

### Testing

**Integration test with specific PDF:**
```bash
./src/test/test_workflow.sh -i path/to/paper.pdf -v
```

**Test model verification (offline):**
```bash
python src/test/test_model_verification.py -v
```

**Quick workflow test:**
```bash
./test-wf.sh -p 2 -v
```

### Utility Commands

**List configured models:**
```bash
python -m common.list_models -p 4
```

**Review processing status:**
```bash
python -m mcq_workflow.review_status -o _RESULTS
```

**Extract Q&A pairs:**
```bash
python -m mcq_workflow.extract_qa -i input.json -o output.json
```

## Model Configuration

Models are specified in `config.local.yml`:
```yaml
workflow:
  extraction: openai:gpt-4o          # Model for MCQ generation
  contestants: [openai:gpt-4o, argo:mistral-7b]  # Models for answering
  target: openai:gpt-4o              # Target model for fine-tuning
```

Reference models by `shortname` or `provider:model` format. Add new endpoints in `servers.yml` with corresponding credentials in `secrets.yml`.

## Key Implementation Details

- **Parallel Processing**: All major operations support `-p` parameter for concurrent execution
- **Chunk-based Processing**: Papers are split into 1000-token chunks (configurable in config.yml)
- **Error Handling**: Verbose mode (`-v`) exposes detailed operation logs
- **Resume Capability**: Workflow can restart from specific steps using `--step` parameter
- **Agent Architecture**: New agent-based system in `agent/` directory provides alternative orchestration

## Testing Strategy

- **Integration Tests**: `test_workflow.sh` tests complete pipeline with real PDFs
- **Unit Tests**: `test_model_verification.py` validates stub model responses
- **Offline Testing**: Stub model (`test_model.py`) enables development without API calls
- Always use `-v` flag during development for detailed debugging output