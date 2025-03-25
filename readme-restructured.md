# Code for Creating and Scoring Multiple Choice Questions (MCQs) from Papers

## Overview

This repository provides Python programs for:
* Generating and evaluating Multiple Choice Questions (MCQs)
* Fine-tuning models based on supplied data
* Workflow tools for scientific paper analysis

**Contact:** Please email {foster|stevens|catlett}@anl.gov if you see things that are unclear or missing.

## Prerequisites

### ALCF Inference Service Setup
Before you start, we recommend following the instructions for [ALCF Inference Service Prerequisites](https://github.com/argonne-lcf/inference-endpoints?tab=readme-ov-file#%EF%B8%8F-prerequisites) to set up your ALCF authentication token, which is required to access models via the inference service.

### Repository Setup

1. **Clone the Repository:**
```bash
git clone git@github.com:auroraGPT-ANL/MCQ-and-SFT-code.git
cd MCQ-and-SFT-code
```

2. **Prepare Working Directories:**
```bash
mkdir _PAPERS _JSON _MCQ _RESULTS
```

3. **Set Up Conda Environment:**
```bash
# Option 1: Update existing environment
conda env update --name <your_conda_env> --file environment.yml

# Option 2: Create new environment
conda env create -f environment.yml
conda activate globus_env
```

### Workflow Overview

This pipeline converts scientific papers in PDF format into JSON and then uses AI models to:
* Generate multiple-choice questions (MCQs)
* Create answers to those MCQs
* Score the generated answers

**Workflow Steps:**
[View Workflow Flowchart](https://github.com/auroraGPT-ANL/MCQ-and-SFT-code/blob/CeC/MCQ-Workflow.png)

1. Convert PDFs to JSON representations
2. Generate MCQs from JSON files
3. Combine MCQ JSON files
4. Select a subset of MCQs (optional)
5. Generate additional answers for MCQs
6. Score AI-generated answers
7. Review MCQ generation and scoring status

## Workflow Execution

### Bundled Workflow Execution

For a quick and comprehensive run of the entire workflow:

1. Define up to 4 models in `config.yml`
2. Run the bundled workflow script:

```bash
# Run with default 8-way parallelism
./src/run_workflow.sh

# Run with 12-way parallelism
./src/run_workflow.sh -p 12

# Run with 20 randomly selected MCQs
./src/run_workflow.sh -n 20
```

### Detailed Step-by-Step Workflow

#### 1. Convert PDFs to JSON
```bash
# Default parsing
python src/simple_parse.py

# Specify input and output directories
python src/simple_parse.py -i _PAPERS -o _JSON
```

#### 2. Generate MCQs
```bash
# Authenticate with ALCF inference service
python src/inference_auth_token.py authenticate

# Generate MCQs (using default or specified model)
python src/generate_mcqs.py
# OR
python src/generate_mcqs.py -m 'alcf:mistralai/Mistral-7B-Instruct-v0.3'
```

#### 3. Combine MCQ JSON Files
```bash
python src/combine_json_files.py -o MCQ-combined.json
```

#### 4. Select MCQ Subset (Optional)
```bash
python src/select_mcqs_at_random.py -i MCQ-combined.json -o MCQ-subset.json -n 17
```

#### 5. Generate Answers
```bash
# Using model from config.yml
python src/generate_answers.py -i MCQ-subset.json

# Specify a different model
python src/generate_answers.py -i MCQ-subset.json -m 'alcf:meta-llama/Meta-Llama-3-70B-Instruct'
```

#### 6. Score Answers
```bash
# Using models from config.yml
python src/score_answers.py

# Specify models explicitly
python src/score_answers.py -a 'model-A' -b 'model-B'
```

#### 7. Review Status
```bash
python src/review_status.py -i MCQ-combined.json
```

## Configuration

### config.yml

The `config.yml` file allows you to:
- Specify default directories
- Define models for each workflow stage
- Set parallelization options

Options include:
- Model selection
- Parallel processing configuration
- Verbosity levels
  - `-v / --verbose`: Show detailed progress messages
  - `-q / --quiet`: Suppress output
  - Default: Progress bar

## Notes

- Authenticate periodically with the ALCF inference service
- Check model availability before running extensive workflows
- Adjust parallelization and model selection as needed

## Additional Resources

- [ALCF Inference Service Documentation](https://github.com/argonne-lcf/inference-endpoints)
- [AdaParse Parser (Alternative PDF Parsing)](https://github.com/7shoe/AdaParse/tree/main)
