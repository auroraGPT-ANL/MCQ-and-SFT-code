# Code for Creating Scientific Training Data from Papers

## Overview
This repository provides Python programs for creating training data to fine-tune models using
scientific papers.  There are two workflows implemented (or being implemented) here.  The first, 
**Multiple Choice Question (MCQ) Workflow**, does the following:
1.  Converts PDF-format papers into JSON
2.  Uses an AI model to generate Multiple Choice Questions (MCQs) for each paper.  Each paper is split into n-token *chunks*, and the model creates an MCQ for each chunk.
3.  Uses one or more models to answer the MCQs
4.  All models used score answers from all other models.

A second workflow, **New Knowledge Nugget (NKN) Workflow**, still under construction, will 
1.  Convert PDF-format papers into JSON
2.  Use an AI model to extract Knowlege Nuggets from each paper.  Each paper is split into n-token *chunks*, and the will extract knowledge nuggets from each.
3.  Test each nugget using a model to be fine-tuned, eliminating nuggets that are already known to the model. This will create a set of *New* Knowledge Nuggets (NKNs) for fine-tuning the target model.

The current, stable mcq\_workflow system operates from the command line, where each component of the workflow can be run as a stand-alone tool or as part of a shell script, 'legacy/scripts/run\_mcq\_workflow.sh'. This script implements the workflow as illustrated in 
[this flowchart](https://github.com/auroraGPT-ANL/MCQ-and-SFT-code/blob/CeC/MCQ-Workflow.png). If only interested in this version, and not tinkering, you may prefer to download the
[Stable-Snapshot: Version-1](https://github.com/auroraGPT-ANL/MCQ-and-SFT-code/releases/tag/Stable-V1)
release (tagged in this repo as
[Stable-V1](https://github.com/auroraGPT-ANL/MCQ-and-SFT-code/tree/Stable-V1)).

### MCQ Workflow Overview

This pipeline converts scientific papers in PDF format into JSON and then uses AI models to:
* Generate multiple-choice questions (MCQs)
* Create answers to those MCQs
* Score the generated answers

**Step-by-Step Workflow:**
[View Example Workflow Flowchart](https://github.com/auroraGPT-ANL/MCQ-and-SFT-code/blob/CeC/MCQ-Workflow.png)
(This chart is a simplified instance of the workflow)

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
* *src/mcq\_workflow* - tools specific to generating, answering, and scoring MCQs, 
* *src/nugget\_workflow* - tools specific to extracting knowledge nuggets and screening for those not already know by a target model,
* *src/tune\_workflow* - tools to take MCQs (and eventually NKNs) to fine-tune a model. (also under construction, thus not yet included in either workflow)
* *src/test* - test routines including a stub model for testing workflows quickly without model delays (including offline testing), and
* *legacy/scripts* shell script to execute workflow (replaced with a python script in *src/mcq\_workflow*).

**Contact:** Please email {foster|stevens|catlett}@anl.gov if you see things that are unclear or missing.

---
### Clone the Repo and Get Set up

1. **Clone the Repository:**
```bash
git clone git@github.com:auroraGPT-ANL/MCQ-and-SFT-code.git
cd MCQ-and-SFT-code
```
*Alternatively*... if you are not using SSH access to Github, you can:
```bash
git clone https://github.com/auroraGPT-ANL/MCQ-and-SFT-code.git
cd MCQ-and-SFT-code
```

2. **Prepare Working Directories:**
```bash
mkdir _PAPERS _JSON _MCQ _RESULTS
```

3. **Set Up Conda Environment:**

Option 1: Update your existing Conda environment
```bash
conda env update --name YOUR_CONDA_ENV --file environment.yml
```

Option 2: Create new environment
```bash
conda env create -f environment.yml
conda activate augpt_env
```

4. Add the (full, absolute paths) to the src directories to your PYTHONPATH:
```bash
export PYTHONPATH="$HOME/MCQ-and-SFT-code:$HOME/MCQ-and-SFT-code/src${PYTHONPATH:+:$PYTHONPATH}"
```
To avoid having to do this every time you activate the conda env, add this to your *~/.zshrc* or 
*~/.bashrc*::
```bash
# set PYTHONPATH for MCQ pipeline at MCQ-and-SFT-code
export PYTHONPATH="$HOME/MCQ-and-SFT-code:$HOME/YOUR_PATH/MCQ-and-SFT-code/src${PYTHONPATH:+:$PYTHONPATH}"
```

Note- Make sure to update **YOUR**\_**PATH**.

5. **Populate \_PAPERS:** Place PDF-formatted input materials (e.g., scientific papers) into \_PAPERS. (Note 
this workflow is only processing text)

6. **Set up configuration:** Edit *config.yml* to specify at least two and up to four
models you wish to use.  (see **Configuration** notes below)


### Configuration-file guide

| File | Tracked? | Purpose |
|------|----------|---------|
| `config.yml` | ✓ | Stable defaults: prompts, directory names, CLI defaults. |
| `servers.yml` | ✓ | Shared **endpoint catalogue** — one entry per inference host. |
| `config.local.yml` | ✗ | **Your run-time choices**: extraction / contestant / target models, per-user tweaks. |
| `secrets.yml` | ✗ | Credentials: API keys, usernames, tokens referenced by `servers.yml`. |

**Override precedence (highest → lowest)**  
`env vars` ▸ `config.local.yml` ▸ `servers.yml` ▸ `config.yml`

### First end-to-end test

1. Create your working dirs

```bash
mkdir -p _PAPERS _JSON _MCQ _RESULTS
```

2. Run the entire workflow with 4 parallel workers with the --verbose option.

This (-v --verbose) option exposes details of anything that might go wrong.
Once you are running smoothly, it's prettier without the --verbose option.

```bash
python -m mcq_workflow.run_workflow -p 4 -v
```

The script loads models, generates MCQs, answers, and scores.

---

## Configuration reference 

`config.yml`
```yaml
directories:
  papers: _PAPERS
  json_dir: _JSON
  mcq: _MCQ
  results: _RESULTS
quality:
  chunkSize: 1000
  defaultThreads: 4
prompts:
  mcq_system: |
    You are a helpful assistant…
```

`config.local.yml` (example)
```yaml
workflow:
  extraction: openai:gpt-4o
  contestants: [openai:gpt-4o, argo:mistral-7b]
  target: openai:gpt-4o      # optional for future fine-tuning
timeout: 60
default_temperature: 0.7
```

`servers.yml`
```yaml
oak:              # shortname users reference in CLI
  shortname: oak
  provider: openai
  base_url: https://api.openai.com/v1
  model: gpt-4o
  cred_key: openai_api_key
argo_dev:
  shortname: argo
  provider: argo
  base_url: https://argo-dev.alcf.anl.gov/v1
  model: mistralai/Mistral-7B-Instruct-v0.3
  cred_key: argo_token
```

`secrets.yml`
```yaml
openai_api_key:   "sk-…"
argo_token:       "Bearer abcdef…"
```

---

## FAQ (new)

* **Q: I changed models in `config.local.yml` but the run still calls the old ones.**  
  A: Make sure you didn’t also set `AUGPT_WORKFLOW__…` environment variables; env vars override YAML.

* **Q: How do I add a brand-new endpoint?**  
  1. Add a block in `servers.yml` with `provider`, `base_url`, `model`, `cred_key`.  
  2. Add the credential in `secrets.yml`.  
  3. Reference its `shortname` (or `provider:model`) in `config.local.yml`.

* **Q: Where did the old `-m` flags go?**  
  A: They still work, but the recommended way is to list models in `config.local.yml` so the whole workflow is reproducible.

---

## More Detailed Overview

This overview may have some errors as the update in this branch is a WIP.


*Last updated: 2025-05-25*

