# Code for Creating Scientific Training Data from Papers

## Overview

This repository provides Python programs for creating training data to fine-tune models using scientific papers. There are two workflows implemented (or being implemented) here:

**Multiple Choice Question (MCQ) Workflow** does the following:
1. Converts PDF-format papers into JSON
2. Uses an AI model to generate Multiple Choice Questions (MCQs) for each paper. Each paper is split into n-token *chunks*, and the model creates an MCQ for each chunk.
3. Uses one or more models to answer the MCQs
4. All models used score answers from all other models.

**New Knowledge Nugget (NKN) Workflow** (under construction) will:
1. Convert PDF-format papers into JSON
2. Use an AI model to extract Knowledge Nuggets from each paper. Each paper is split into n-token *chunks*, and the will extract knowledge nuggets from each.
3. Test each nugget using a model to be fine-tuned, eliminating nuggets that are already known to the model. This will create a set of *New* Knowledge Nuggets (NKNs) for fine-tuning the target model.

The current system operates from the command line, where each component of the workflow can be run as a stand-alone tool or as part of the main workflow orchestrator.

### MCQ Workflow Overview

This pipeline converts scientific papers in PDF format into JSON and then uses AI models to:
* Generate multiple-choice questions (MCQs)
* Create answers to those MCQs
* Score the generated answers

**Step-by-Step Workflow:**
1. Convert PDFs to JSON representations
2. Generate MCQs from JSON files
3. Combine MCQ JSON files
4. Select a subset of MCQs (optional)
5. Generate additional answers for MCQs
6. Score AI-generated answers
7. Review MCQ generation and scoring status

### This Repo

The repository is organized as follows:

* *src/common* - tools common to both the MCQ and Nugget workflows, including model access, configuration, etc., 
* *src/mcq_workflow* - tools specific to generating, answering, and scoring MCQs, 
* *src/nugget_workflow* - tools specific to extracting knowledge nuggets and screening for those not already know by a target model,
* *src/tune_workflow* - tools to take MCQs (and eventually NKNs) to fine-tune a model. (also under construction, thus not yet included in either workflow)
* *src/test* - test routines including a stub model for testing workflows quickly without model delays (including offline testing), and
* *legacy/scripts* shell script to execute workflow (replaced with a python script in *src/mcq_workflow*).

**Contact:** Please email {foster|stevens|catlett}@anl.gov if you see things that are unclear or missing.

---

## Quick Start Guide

### 1. Clone and Setup Environment

**Clone the Repository:**
```bash
git clone https://github.com/auroraGPT-ANL/MCQ-and-SFT-code.git
cd MCQ-and-SFT-code
```

**Create Conda Environment:**
```bash
conda env create -f environment.yml
conda activate augpt_env
```

**Set Python Path (Required):**
```bash
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

**Create Working Directories:**
```bash
mkdir -p _PAPERS _JSON _MCQ _RESULTS
```

### 2. Configuration Setup

The system uses 4 configuration files with this precedence (highest → lowest):  
`env vars` ▸ `config.local.yml` ▸ `servers.yml` ▸ `config.yml`

| File | Tracked? | Purpose | Action Needed |
|------|----------|---------|---------------|
| `config.yml` | ✓ | Stable defaults | Already exists |
| `servers.yml` | ✓ | Endpoint catalog | Already exists |
| `config.local.yml` | ✗ | **Your models & settings** | **You must create** |
| `secrets.yml` | ✗ | **Your API keys** | **You must create** |

#### Create Your Local Configuration

**Step 2a: Create `config.local.yml`**
```bash
# Create your local configuration file
cat > config.local.yml << 'EOF'
# Your model choices for this run
workflow:
  extraction: openai:gpt-4o-mini     # Model for generating MCQs
  contestants: [openai:gpt-4o-mini]  # Models for answering MCQs
  target: openai:gpt-4o-mini         # Target model (optional)

# Optional: Override defaults
timeout: 60
default_temperature: 0.7
EOF
```

**Step 2b: Create `secrets.yml`**
```bash
# Create your secrets file (never commit this!)
cat > secrets.yml << 'EOF'
# Add your API credentials here
openai_api_key: "sk-your-openai-key-here"

# Uncomment and add other credentials as needed:
# argo_username: "your-argo-username"
# alcf_token: "your-alcf-token"
EOF
```

⚠️ **Important**: Replace `"sk-your-openai-key-here"` with your actual OpenAI API key.

### 3. Test Your Setup

**Verify Configuration:**
```bash
python test_settings_basic.py
```

**Add Sample Papers:**
```bash
# Place PDF files in _PAPERS directory
cp /path/to/your/papers/*.pdf _PAPERS/
```

**Run Quick Test:**
```bash
# Test the complete workflow with minimal settings
python -m mcq_workflow.run_workflow -p 2 -v
```

### 4. Understanding the Configuration System

#### Available Model Endpoints

Check `servers.yml` to see available endpoints. Common patterns:

- **OpenAI**: `openai:gpt-4o-mini`, `openai:gpt-4o`
- **Local servers**: Use shortnames like `scout`, `qwen`  
- **ALCF**: `alcf:meta-llama/Meta-Llama-3-70B-Instruct`
- **Test models**: `test:all` (for offline development)

#### Model Configuration Examples

**Simple OpenAI setup:**
```yaml
workflow:
  extraction: openai:gpt-4o-mini
  contestants: [openai:gpt-4o-mini]
```

**Multiple models:**
```yaml
workflow:
  extraction: openai:gpt-4o
  contestants: [openai:gpt-4o, openai:gpt-4o-mini, scout]
  target: openai:gpt-4o
```

**Offline testing:**
```yaml
workflow:
  extraction: test:all
  contestants: [test:all]
```

#### Adding New Endpoints

1. **Add endpoint to `servers.yml`:**
```yaml
my_endpoint:
  shortname: my_model
  provider: openai  # or: argo, alcf, local, hf
  base_url: https://api.example.com/v1
  model: my-model-name
  cred_key: my_api_key
```

2. **Add credential to `secrets.yml`:**
```yaml
my_api_key: "your-secret-key"
```

3. **Reference in `config.local.yml`:**
```yaml
workflow:
  extraction: my_model  # or: openai:my-model-name
```

### 5. Troubleshooting Setup

**Configuration not working?**
```bash
# Check file exists and format
ls -la config.local.yml secrets.yml
python -c "import yaml; print(yaml.safe_load(open('config.local.yml')))"
```

**Import errors?**
```bash
# Ensure PYTHONPATH is set
echo $PYTHONPATH
conda activate augpt_env
```

**API errors?**
```bash
# Test credentials
python -c "
from common.loader import load_settings
settings = load_settings()
print('Available endpoints:', list(settings.endpoints.keys()))
print('Configured models:', settings.workflow.contestants)
"
```

---

## Detailed Workflow Commands

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

---

## Authentication Setup

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

---

## FAQ

* **Q: I changed models in `config.local.yml` but the run still calls the old ones.**  
  A: Make sure you didn't also set `AUGPT_WORKFLOW__…` environment variables; env vars override YAML.

* **Q: How do I add a brand-new endpoint?**  
  1. Add a block in `servers.yml` with `provider`, `base_url`, `model`, `cred_key`.  
  2. Add the credential in `secrets.yml`.  
  3. Reference its `shortname` (or `provider:model`) in `config.local.yml`.

* **Q: Where did the old `-m` flags go?**  
  A: They still work, but the recommended way is to list models in `config.local.yml` so the whole workflow is reproducible.

---

*Last updated: 2025-07-01*