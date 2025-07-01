# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository provides Python programs for creating scientific training data from papers through two main workflows:

1. **Multiple Choice Question (MCQ) Workflow**: Converts PDF papers to JSON, generates MCQs from each paper chunk, creates answers using multiple AI models, and scores all answers.
2. **New Knowledge Nugget (NKN) Workflow**: Extracts knowledge nuggets from papers and filters for novel information not already known to target models (under development).

The repository supports both a stable CLI-based workflow and an experimental agent-based architecture.

## Architecture

### Core Components

**Model Access Layer** (`src/common/model_access.py`):
- Unified interface for multiple LLM backends (OpenAI, ALCF, Argo, HuggingFace, local vLLM, etc.)
- Model specification format: `provider:model` (e.g., `openai:gpt-4o`) or shortname references
- Thread-safe parallel processing with robust error handling and graceful degradation

**MCQ Workflow** (`src/mcq_workflow/`):
- `generate_mcqs.py`: Three-step MCQ generation (summarize chunk, generate MCQ, verify answer)
- `generate_answers.py`: Parallel answer generation using multiple models
- `score_answers.py`: Cross-model answer scoring for evaluation
- `mcq_util.py`: Core processing logic with intelligent text chunking using spaCy
- `run_workflow.py`: Main workflow orchestrator replacing legacy shell scripts

**Agent Architecture** (`agent/`):
- Agent-based workflow system where each component implements `run(context: dict) -> dict`
- Orchestrator manages agent execution and context passing
- Experimental extension of the CLI workflow

### Configuration System

The configuration uses a layered approach with override precedence:

1. **`config.yml`** (tracked) - Stable defaults: prompts, directories, CLI defaults
2. **`servers.yml`** (tracked) - Shared endpoint catalogue for inference hosts  
3. **`config.local.yml`** (untracked) - Your runtime choices: models, per-user tweaks
4. **`secrets.yml`** (untracked) - Credentials: API keys, tokens referenced by servers.yml

Override precedence: `env vars` → `config.local.yml` → `servers.yml` → `config.yml`

### Directory Structure

- `src/common/`: Shared utilities (model access, config, parsing, authentication)
- `src/mcq_workflow/`: MCQ-specific processing pipeline
- `src/nugget_workflow/`: Knowledge nugget extraction (under development)
- `src/tune_workflow/`: Fine-tuning utilities (LORA, full fine-tune)
- `src/test/`: Test infrastructure including stub models for offline testing
- `agent/`: Experimental agent-based architecture
- `legacy/scripts/`: Original shell script-based workflow (maintained for compatibility)

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

**Configuration Setup:**
```bash
# Copy and edit local configuration
cp config.yml config.local.yml
# Edit config.local.yml to specify your models and settings
```

## Development Commands

### Main Workflow Execution

**Full MCQ workflow with 4 parallel workers:**
```bash
python -m mcq_workflow.run_workflow -p 4 -v
```

**Start from specific step (1-5):**
```bash
python -m mcq_workflow.run_workflow --step 3 -p 4
```

**Process subset of MCQs:**
```bash
python -m mcq_workflow.run_workflow -n 50 -p 4
```

**Specify number of answer choices:**
```bash
python -m mcq_workflow.run_workflow -a 5 -p 4  # Generate 5-choice MCQs
```

### Individual Pipeline Components

**Convert PDFs to JSON:**
```bash
python -m common.simple_parse -i _PAPERS -o _JSON
```

**Generate MCQs:**
```bash
python -m mcq_workflow.generate_mcqs -p 4 -v
python -m mcq_workflow.generate_mcqs -m openai:gpt-4o -a 5  # 5-choice MCQs
```

**Generate answers:**
```bash
python -m mcq_workflow.generate_answers -i MCQ-combined.json -m openai:gpt-4o -p 4
```

**Score answers:**
```bash
python -m mcq_workflow.score_answers -a openai:gpt-4o -b argo:mistral-7b -p 4
```

**Combine MCQ files:**
```bash
python -m common.combine_json_files -o MCQ-combined.json
```

### Testing Commands

**Integration test with specific PDF:**
```bash
./src/test/test_workflow.sh -i path/to/paper.pdf -v
```

**Test model verification (offline testing):**
```bash
python -m test.test_model_verification -v
```

**Legacy workflow test:**
```bash
./legacy/scripts/run_mcq_workflow.sh -p 2 -v
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

**Check ALCF service status:**
```bash
python -m common.check_alcf_service_status
```

**Extract Q&A pairs from results:**
```bash
python -m mcq_workflow.extract_qa -i input.json -o output.json
```

**Select random MCQ subset:**
```bash
python -m common.select_mcqs_at_random -i MCQ-combined.json -o MCQ-subset.json -n 100
```

## Model Configuration

### Model Specification

Models are specified in `config.local.yml`:
```yaml
workflow:
  extraction: openai:gpt-4o          # Model for MCQ generation
  contestants: [openai:gpt-4o, argo:mistral-7b]  # Models for answering
  target: openai:gpt-4o              # Target model for fine-tuning
```

### Supported Model Types

- `openai:` - OpenAI API models (requires API key in secrets.yml)
- `argo:` - Argonne ARGO service (requires username in secrets.yml)
- `alcf:` - ALCF Inference Service (requires ALCF authentication token)
- `hf:` - Local HuggingFace models (requires HF token for private models)
- `local:` - Local vLLM servers
- `test:` - Test stub models for offline development

### Adding New Endpoints

Add to `servers.yml`:
```yaml
new_endpoint:
  shortname: my_model
  provider: openai
  base_url: https://api.example.com/v1
  model: my-model-name
  cred_key: my_api_key
```

Add credentials to `secrets.yml`:
```yaml
my_api_key: "your-secret-key-here"
```

## Testing Strategy

- **Integration Tests**: `test_workflow.sh` tests complete pipeline with real PDFs
- **Unit Tests**: `test_model_verification.py` validates stub model responses  
- **Offline Testing**: Stub model (`test_model.py`) enables development without API calls
- **Legacy Compatibility**: Original shell scripts maintained for comparison
- Always use `-v` flag during development for detailed debugging output

## Authentication

### ALCF Inference Service
```bash
python -m common.inference_auth_token authenticate
python -m common.inference_auth_token get_access_token  # Check current token
```

### OpenAI
Add to `secrets.yml`:
```yaml
openai_api_key: "sk-your-key-here"
```

### Argo
Add to `secrets.yml`:
```yaml
argo_username: "your-argo-username"
```

## Key Implementation Details

- **Parallel Processing**: All major operations support `-p` parameter for concurrent execution
- **Chunk-based Processing**: Papers are split into configurable token chunks (default 1000)
- **Error Handling**: Robust error recovery with configurable retry logic
- **Resume Capability**: Workflow can restart from specific steps using `--step` parameter
- **Progress Tracking**: Optional progress bars and detailed logging with `-v`
- **Configuration-driven**: Prompts, models, and parameters easily modified via YAML files
- **Multi-backend Support**: Seamless switching between different LLM providers

## File Formats

- **Input**: PDF files in `_PAPERS/`
- **Intermediate**: JSON files with parsed text in `_JSON/`, MCQ files in `_MCQ/`
- **Output**: Answer and scoring results in `_RESULTS/` as JSONL files
- All data uses consistent JSON/JSONL schemas with UTF-8 encoding