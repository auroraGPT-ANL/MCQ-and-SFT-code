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

Specifically:

1. Download the script to manage access tokens:
```bash
wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py
```
2. Authenticate with your Globus account:
```bash
python inference_auth_token.py authenticate
```
The above command will generate an access token and a refresh token, and store them in your home directory. 

3. Other Tips

If you need to re-authenticate from scratch in order to 1) change Globus account, or 2) resolve a `Permission denied from internal policies` error, first logout from your account by visiting [https://app.globus.org/logout](https://app.globus.org/logout), and type the following command:
```bash
python inference_auth_token.py authenticate --force
```
View your access token:
```bash
python inference_auth_token.py get_access_token
```
If your current access token is expired, the above command will atomatically generate a new token without human intervention.

> **⏰ Token Validity:** All access tokens are valid for 48 hours, but the refresh token will allow you to acquire new access tokens programatically without needing to re-authenticate. Refresh tokens do not expire unless they are left unused for 6 months or more. However, an internal policy will force users to re-authenticate every 7 days.
> 
> **🔒 Access Note:**
> * Endpoints are restricted. You must be on Argonne's network (Use VPN, Dash, or SSH to ANL machine).
> * You will need to authenticate with Argonne or ALCF SSO (Single Sign On) using your credentials.


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
Option 1: Update your existing Conda environment
```bash
conda env update --name <your_conda_env> --file environment.yml
```

Option 2: Create new environment
```bash
conda env create -f environment.yml
conda activate globus_env
```

4. **Populate \_PAPERS:** Place PDF-formatted input materials (e.g., scientific papers) into \_PAPERS.

5. **Set up configuration:** Edit *config.yml* to specify at least two and up to four
models you wish to use.  (see **Configuration** notes below)

### Workflow Overview

This pipeline converts scientific papers in PDF format into JSON and then uses AI models to:
* Generate multiple-choice questions (MCQs)
* Create answers to those MCQs
* Score the generated answers

**Step-by-Step Workflow:**
[View Workflow Flowchart](https://github.com/auroraGPT-ANL/MCQ-and-SFT-code/blob/CeC/MCQ-Workflow.png)

1. Convert PDFs to JSON representations
2. Generate MCQs from JSON files
3. Combine MCQ JSON files
4. Select a subset of MCQs (optional)
5. Generate additional answers for MCQs
6. Score AI-generated answers
7. Review MCQ generation and scoring status

## Workflow Execution

Note that if you are using the ALCF inference endpoint service you might first check to see if any
models are running, as it takes 10-15 minutes for a model to load.  This will cause any of the 
codes below (generate\_mcqs.py, generate\_answers.py, score\_answers.py) to time out.

**Check which models are running**

You may wish to check to see which models are currently running as waiting for a model to load can
take 10-15 minutes (see 
[ALCF Inference service](https://github.com/argonne-lcf/inference-endpoints)). Get the list of running
and queued models as follows:
   ```bash
   access_token=$(python src/inference_auth_token.py get_access_token)
   curl -X GET "https://data-portal-dev.cels.anl.gov/resource_server/sophia/jobs" \
       -H "Authorization: Bearer ${access_token}" | jq
   ```
Piping the output to ``jq`` (Command-line JSON processor) makes it much easier to read.

**Notes**
 - If you are not connected via VPN or to Argonne-auth at the lab then you'll get an error such as *curl: (6) Could not resolve host: data-portal-dev.cels.anl.gov*.
 - If it's been a while since you authenticated, you'll get a "Permission denied" error. In this case, you'll need to re-authenticate:
```
python src/inference_auth_token.py authenticate --force
```




### Bundled Workflow Execution

For a quick and comprehensive run of the entire workflow:

1. Define up to 4 models in `config.yml`
2. Run the bundled workflow script:

**Examples:**

Run with default 8-way parallelism
```bash
./src/run_workflow.sh
```

Run with 12-way parallelism
```bash
./src/run_workflow.sh -p 12
```

Run with 20 randomly selected MCQs
```bash
./src/run_workflow.sh -n 20
```

### Detailed Step-by-Step Workflow

#### 1. Convert PDFs to JSON
Default parsing
```bash
python src/simple_parse.py
```

or explicitly specify input and output directories
```bash
python src/simple_parse.py -i _PAPERS -o _JSON
```

#### 2. Generate MCQs
Authenticate with ALCF inference service
```bash
python src/inference_auth_token.py authenticate
```

Generate MCQs (using default or specified model)
```bash
python src/generate_mcqs.py
```
or to specify a different model than in *config.yml*:
```bash
python src/generate_mcqs.py -m 'alcf:mistralai/Mistral-7B-Instruct-v0.3'
```
(noting that models are specified as *location*:*model_name* - see **Additional Notes** below)

#### 3. Combine MCQ JSON Files
```bash
python src/combine_json_files.py -o MCQ-combined.json
```

#### 4. Select MCQ Subset (Optional)
```bash
python src/select_mcqs_at_random.py -i MCQ-combined.json -o MCQ-subset.json -n 17
```

#### 5. Generate Answers
Using model from config.yml
```bash
python src/generate_answers.py -i MCQ-subset.json
```
Or to specify a different model than the one in *config.yml*:
```bash
python src/generate_answers.py -i MCQ-subset.json -m 'alcf:meta-llama/Meta-Llama-3-70B-Instruct'
```

#### 6. Score Answers
Using models from config.yml
```bash
python src/score_answers.py
```

Or to specify models explicitly (different than the ones in *config.yml*:
```bash
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


### Below this point the paths, etc. are outdated and need to be fixed

## Additional Notes
- This pipeline ensures **high-quality multiple-choice questions** are generated and scored using AI.
- The steps allow for **comparison of AI-generated answers against reference answers**.
- The scoring step provides a **numerical evaluation (1-10)** of answer accuracy.

**Note:**
* You need a file *openai_access_token.txt* that contains your OpenAI access token if you
are to use an OpenAI model like *gpt-4o*.

Examples of running *generate_answers.py*:
* `python src/generate_answers.py -o ../_RESULTS -i ../_MCQ -m openai:o1-mini.json`
  * Uses the OpenAI model `o1-mini` to generate answers for MCQs in `MCQs.json` and stores results in the `_RESULTS` directory, in a file named `answers_openai:o1-mini.json`
* `python src/generate_answers.py -o ../_RESULTS -i MCQs.json -m "pb:argonne-private/AuroraGPT-IT-v4-0125`
  * Uses the Huggingface model `argonne-private/AuroraGPT-IT-v4-0125`, running on a Polaris compute node started via PBS, to generate answers for the same MCQs. Results are placed in `_RESULTS/answers_pb:argonne-private+AuroraGPT-IT-v4-0125.json`
 
Examples of running `score_answers.py`:
* `python score_answers.py -o _RESULTS -i MCQs.json -a openai:o1-mini.json -b openai:gpt-4o`
  * Uses the OpenAI model `gpt-4o` to score answers for MCQs in `MCQs.json` and stores results in `_RESULTS` directory, in a file named `answers_openai:o1-mini.json`
* `python score_answers.py -o _RESULTS -a pb:argonne-private/AuroraGPT-IT-v4-0125 -b openai:gpt-4o`
  * Uses the OpenAI model gpt-4o to score answers previously generated for model `pb:argonne-private/AuroraGPT-IT-v4-0125`, and assumed to be located in a file `_RESULTS/answers_pb:argonne-private+AuroraGPT-IT-v4-0125.json`, as above. Places results in file `_RESULTS/scores_pb:argonne-private+AuroraGPT-IT-v4-0125:openai:gpt-4o.json`.
 

## Notes on different model execution locations
The class `Model` (in `model_access.py`) implements init and run methods that allow for use of different models. 
```
model = Model(modelname)
response = model.run(user_prompt='Tell me something interesting')
```
where `modelname` has a prefix indicating the model type/location:
* **alcf**: Model served by the ALCF Inference Service. You need an ALCF project to charge to.
* **hf**: Huggingface model downloaded and run on Polaris login node (not normally a good thing).
* **pb**: Huggingface model downloaded and run on a Polaris compute node. You need an ALCF project to charge to.
* **vllm**: Huggingface model downloaded and run via VLLM on Polaris compute node. Not sure that works at present.
* **openai**: An OpenAI model, like gpt-4o or o1-mini. You need an OpenAI account to charge to.


## Code for fine-tuning programs
```
# LORA fine-tuning
python lora_fine_tune.py -i <json-file> -o <model-directory>

# Full fine tune
python full_fine_tune.py -i <json-file> -o <model-directory>
```
Note:
* You need a file `hf_access_token.txt` if you want to publish models to HuggingFace.
* You need to edit the file to specify where to publish models in HuggingFace
* We are still debugging how to download and run published models

## Code for other useful things

Determine what models are currently running on ALCF inference service (see below for more info)
```
python check_alcf_service_status.py
```
Determine what answers have been generated and scored, and what additional runs could be performed, _given running models_, to generate and score additional answers. (You may want to submit runs to start models. Use `-m` flag to see what could be useful to submit.) 
```
python review_status.py -o <result-directory>
```
Perform runs of `generate_answers` and `grade_answers.py` to generate missing outputs. (See below for more info)
```
python run_missing_generates.py -o <result-directory>
```

### More on `check_alcf_service_status.py` 

The program `check_alcf_service_status.py` retrieves and processes status information from the
[ALCF Inference service](https://github.com/argonne-lcf/inference-endpoints),
and lists models currently running or queued to run. E.g., as follows, which shows six
models running and one queued. Models that are not accessed for some period are shut
down and queued models started. A request to a model that is not running adds it to the queue.
```
% python check_alcf_service_status.py
Running: ['meta-llama/Meta-Llama-3-70B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.3']
Starting: ['N/A']
Queued : []
```
Note:
* You need a valid ALCF access token stored in a file `alcf_access_token.txt`.  See [how to generate an ALCF access token](https://github.com/argonne-lcf/inference-endpoints?tab=readme-ov-file#authentication).
* Here is a list of [models supported by the ALCF inference service](https://github.com/argonne-lcf/inference-endpoints?tab=readme-ov-file#-available-models).
* "N/A" is a test model used by ALCF, it can be ignored.

### More on `run_missing_generates.py`

The ALCF inference service hosts many models, as [listed here](https://github.com/argonne-lcf/inference-endpoints?tab=readme-ov-file#-available-models). At any one time, zero or more *running*, zero or more are *queued*, and the rest are neither running not queued. (See below for how to use `check_alcf_service_status.py` to determine which.)
You may want to run against all available models. To do so, you can specify `-a all`, which works out what commands are needed to process specified MCQs with all *running models*. Adding `-q` also considers *queued models*, and `-s` *non-running models*. For example, when I ran the following command I was informed of the commands to run three models for which results are not found:
```
% python run_missing_generates.py -i 100-papers-qa.json -o output_files -a all -m 100 -s
python generate_and_grade_answers.py -i 100-papers-qa.json -o outputs -a 'Qwen/Qwen2-VL-72B-Instruct' -b 'gpt-4o' -c -q -s 0 -e 100
python generate_and_grade_answers.py -i 100-papers-qa.json -o outputs -a 'deepseek-ai/DeepSeek-V3' -b 'gpt-4o' -c -q -s 0 -e 100
python generate_and_grade_answers.py -i 100-papers-qa.json -o outputs -a 'mgoin/Nemotron-4-340B-Instruct-hf' -b 'gpt-4o' -c -q -s 0 -e 100
```

`run_missing_generates.py` has options as follows:

```
  -h, --help            show this help message and exit
  -a MODELA, --modelA MODELA
                        modelA
  -o OUTPUTDIR, --outputdir OUTPUTDIR
                        Directory to look for run results
  -i INPUTFILE, --inputfile INPUTFILE
                        File to look for inputs
  -x, --execute         Run program
  -q, --queued          Process queued models
  -m MAX, --max MAX     Max to process
  -s, --start           Request to non-running models
```




