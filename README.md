# Code for Creating Scientific Training Data from Papers

## Overview
*(unchanged — details redacted for brevity)*

---

## Quick-start for a Fresh Clone <!-- NEW SECTION -->

```bash
git clone https://github.com/auroraGPT-ANL/MCQ-and-SFT-code.git
cd MCQ-and-SFT-code

# 1️⃣  Create/activate a clean Conda env
conda env create -f environment.yml      # or update --name EXISTING_ENV
conda activate augpt_env

# 2️⃣  Install project requirements
pip install -r requirements.txt

# 3️⃣  Generate **user-specific** config & secrets skeletons  🔑
python -m common.init_config             # creates config.local.yml & secrets.yml
```

> **Now open `config.local.yml` and list the models you want to use**  
> (e.g. `extraction: openai:gpt-4o`, `contestants: [openai:gpt-4o, argo:mistral-7b]`).  
> Put any API keys or tokens in **secrets.yml** — never in the tracked files.

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

```bash
# make working dirs
mkdir -p _PAPERS _JSON _MCQ _RESULTS

# drop a few PDF papers into _PAPERS …

# run the entire workflow with 4 parallel workers
python -m mcq_workflow.run_workflow -p 4 -v
```

The script loads models, generates MCQs, answers, and scores — watch the progress bars.

---

*(everything below — MCQ Workflow Overview, detailed step-by-step, ALCF notes, etc. — remains exactly as in your current README; no changes needed)*

---

## Configuration reference <!-- UPDATED SECTION -->

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

*Last updated: 2025-05-24*

