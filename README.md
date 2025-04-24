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

Finally, this repo contains a work-in-progress, exploratory project to use the components from these two workflows as part of an **agentic systems**.

The repository is thus organized as follows:

1. Stable MCQ workflow in *legacy/scripts* uses components in *src* including:
* *src/common* - tools common to both the MCQ and Nugget workflows, including model access, configuration, etc., 
* *src/mcq\_workflow* - tools specific to generating, answering, and scoring MCQs, 
* *src/nugget\_workflow* - tools specific to extracting knowledge nuggets and screening for those not already know by a target model,
* *src/test* - test routines including a stub model for testing workflows quickly without model delays (including offline testing), and
* *src/tune\_workflow* - tools to take MCQs (and eventually NKNs) to fine-tune a model. (also under construction, thus not yet included in either workflow)

All of the components in *src/common* *src/mcq\_workflow* and *src/nugget\_workflow* work both as Python
modules (called form the CLI) and as part of an exploratory agent-based system, where each pipeline component is a
subclass of *agent\_base.Agent* which enforces a python contract of the form:
```
def run(context: dict) -> dict 
```
Each component of the pipeline performs its specific set of tasks and returns its results to a shared *context* A light-weight *orchestrator.py* imports and runs the agents.

The remainder of this README is currently specific to the CLI (legacy, stable) MCQ workflow.

**Contact:** Please email {foster|stevens|catlett}@anl.gov if you see things that are unclear or missing.

## MCQ Workflow

### Table of Contents

- [MCQ Workflow Overview](#workflow-overview)
- [MCQ Workflow Execution](#workflow-execution)
- [Configuration](#configuration)
- [Models](#models)
- [Notes](#notes)


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

Important note: this install has been tested on MacOS (15.4) running on Apple M2 silicon. On 
oher platforms the specifics in *environment.yml* may need to be tweaked.

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

At minimum you need to specify one, ideally two models. Follow the instructions in **Configuration** below.

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

## Workflow Execution

Note that if you are using the ALCF inference endpoint service you might first check to see if any
models are running, as it takes 10-15 minutes for a model to load.  This will cause any of the 
codes below (generate\_mcqs.py, generate\_answers.py, score\_answers.py) to time out.

### Bundled MCQ Workflow Execution

For a quick and comprehensive run of the entire
[MCQ workflow:](https://github.com/auroraGPT-ANL/MCQ-and-SFT-code/blob/CeC/MCQ-Workflow.png)

1. Define up to 4 models in `config.yml`
2. Run the bundled workflow script. 

By default, the workflow script (and each individual python script) displays a progress bar
for each step.  The input and output directories default to those in 'config.yml' but can 
be overridden on the command line with -i and/or -o options.
Similarly, 'config.yml' specifies up to four models to use.  The first one (Model\_A) is
the default model used for the 'generate\_mcqs' script.  All models (including Model\_A)
are then used to 'generate\_answers.'  The workflow then has each model 'score\_answers'
produced by all other models.  

When running individual python scripts, the default models specified in config.yml can be
overridden on the command line with the -m option.

When the MCQ workflow script is running multiple instances of
the 'generate\_answers' or 'score\_answers' steps, these are run in parallel on your
machine.

The -p option allows you to further parallelize the individual scripts ('generate\_mcqs', 
'generate\_answers', and 'score\_answers'), as these process individual chunks of files.
This option then creates *p* seperate interactions with the model's inference endpoint.
Keep in mind that endpoint services have different limits, but most should have no
trouble with -p up to perhaps ~20.

**Examples:**

The core steps of the workflow access models on external hosts.  You will need to 
specify the models you will use in *config.yml*.  To learn more about specifying
models, see **Models** below.

**TO-DO: Set up a starter config with models that require the first time user to provide an API key (creating secrets.yml) to the model service, such as OpenAI, so that they do not have to stop here and learn all about models and configuration.**

Run with default 8-way parallel, in verbose mode to see progress messages
```bash
./legacy/scripts/run_mcq_workflow.sh -v
```

Run with 16-way parallel
```bash
./legacy/scripts/run_mcq_workflow.sh -p 16
```

Run with 20 randomly selected MCQs
```bash
./legacy/scripts//run_mcq_workflow.sh -n 20
```

### Detailed Step-by-Step Workflow

#### 1. Convert PDFs to JSON
Default parsing
```bash
python -m common.simple_parse
```

or explicitly specify input and output directories
```bash
python -m common.simple_parse -i _PAPERS -o _JSON
```

#### 2. Generate MCQs
**If using the ALCF inference endpoints** (skip if not), authenticate with:
```bash
python -m common.inference_auth_token authenticate
```

Generate MCQs (using default or specified model)
```bash
python -m mcq_workflow.generate_mcqs
```
or to specify a different model than in *config.yml*:
```bash
python -m mcq_workflow.generate_mcqs -m 'alcf:mistralai/Mistral-7B-Instruct-v0.3'
```
(noting that models are specified as *location*:*model_name* - see **Configuration** below)

#### 3. Combine MCQ JSON Files
```bash
python -m common.combine_json_files -o MCQ-combined.json
```

#### 4. Select MCQ Subset (Optional)
```bash
python -m common.select_mcqs_at_random -i MCQ-combined.json -o MCQ-subset.json -n 17
```

#### 5. Generate Answers
Using model from config.yml
```bash
python -m mcq_workflow.generate_answers -i MCQ-subset.json
```
Or to specify a different model than the one in *config.yml*:
```bash
python -m mcq_workflow.generate_answers -i MCQ-subset.json -m 'alcf:meta-llama/Meta-Llama-3-70B-Instruct'
```

#### 6. Score Answers
Using models from config.yml
```bash
python -m mcq_workflow.score_answers
```

Or to specify models explicitly (different than the ones in *config.yml*:
```bash
python -m mcq_workflow.score_answers -a 'model-A' -b 'model-B'
```

#### 7. Review Status
```bash
python -m mcq_workflow.review_status 
```

## Configuration

### config.yml

The `config.yml` file allows you to:
- Specify default directories
- Define models for each workflow stage
- Set parallelization options
- Tweak prompts (carefully)

To run any of these scripts or workflows, you must specify a model using the form *location:model_name*.  
The *location* portion indicates how to access the model (endpoints, etc.), which is handled in 
*common/model_access.py*.  This workflow has been tested with locations *local*, *alcf*, and *argo*.

Where credentials are required, place them in a secrets.yml file (which is .gitignored to avoid accidental
sharing to the world via github).

## General Notes
All scripts generally use the following options and defaults:

Options include:
- Model selection
- Parallel processing configuration
- Verbosity levels
  - `-v / --verbose`: Show detailed progress messages
  - `-q / --quiet`: Suppress output
  - Default: Progress bar

## Models

In 'src/common' you will find the 'model\_access' module, where a number of model *types* are
supported including OpenAI, HF, and Argonne-specific services (ALCF Inference Service, Argo, etc.).
New model types can be added through ths model\_access module.

### ALCF Inference Service Setup
Before you start--if you are using this service--we recommend following the instructions for
[ALCF Inference Service Prerequisites](https://github.com/argonne-lcf/inference-endpoints?tab=readme-ov-file#%EF%B8%8F-prerequisites)
to set up your ALCF authentication token, which is required to access models via the inference service.

Specifically:

1. Authenticate with your Globus account:
```bash
python -m common.inference_auth_token authenticate
```
The above command will generate an access token and a refresh token, and store them in your home directory. 

2. Other Tips

If you need to re-authenticate from scratch in order to 1) change Globus account, or 2) resolve a `Permission denied from internal policies` error, first logout from your account by visiting [https://app.globus.org/logout](https://app.globus.org/logout), and type the following command:
```bash
python -m common.inference_auth_token authenticate --force
```
View your access token:
```bash
python -m common.inference_auth_token get_access_token
```
If your current access token is expired, the above command will atomatically generate a new token without human intervention.

> **⏰ Token Validity:** All access tokens are valid for 48 hours, but the refresh token will allow you to acquire new access tokens programatically without needing to re-authenticate. Refresh tokens do not expire unless they are left unused for 6 months or more. However, an internal policy will force users to re-authenticate every 7 days.
> 
> **🔒 Access Note:**
> * Endpoints are restricted. You must be on Argonne's network (Use VPN, Dash, or SSH to ANL machine).
> * You will need to authenticate with Argonne or ALCF SSO (Single Sign On) using your credentials.


**Working with ALCF inference Endpoints**

Before running the workflow or scripts you should check to see which models are currently
running as waiting for a model to load can take 10-15 minutes (see 
[ALCF Inference service](https://github.com/argonne-lcf/inference-endpoints)).

Get the list of running and queued models as follows:
   ```bash
   access_token=$(python -m common.inference_auth_token get_access_token)
   curl -X GET "https://data-portal-dev.cels.anl.gov/resource_server/sophia/jobs" \
       -H "Authorization: Bearer ${access_token}" | jq
   ```
Piping the output to ``jq`` (Command-line JSON processor) makes it much easier to read.

**Notes**
 - If you are not connected via VPN or to Argonne-auth at the lab then you'll get an error such as *curl: (6) Could not resolve host: data-portal-dev.cels.anl.gov*.
 - If it's been a while since you authenticated, you'll get a "Permission denied" error. In this case, you'll need to re-authenticate:
```
python -m common.inference_auth_token authenticate --force
```

If no models are running, then you'll need to invoke one (and wait 10-15 minutes) using 
one of the codes below (generate\_mcqs.py, generate\_answers.py, score\_answers.py). They will
time out and you'll need to ^C interrupt them, but this will queue up a model to run. 

## Notes

- Authenticate periodically with the ALCF inference service
- Check model availability before running extensive workflows
- Adjust parallelization and model selection as needed

## Additional Resources

- [ALCF Inference Service Documentation](https://github.com/argonne-lcf/inference-endpoints)
- [AdaParse Parser (Alternative PDF Parsing)](https://github.com/7shoe/AdaParse/tree/main)


# Below this point are outdated readme sections that need overhaul 

## Additional Notes
- This pipeline ensures **high-quality multiple-choice questions** are generated and scored using AI.
- The steps allow for **comparison of AI-generated answers against reference answers**.
- The scoring step provides a **numerical evaluation (1-10)** of answer accuracy.

**Note:**
* You need a file *openai_access_token.txt* that contains your OpenAI access token if you
are to use an OpenAI model like *gpt-4o*.

Examples of running *generate_answers.py*:
* `python -m generate_answers -o ../_RESULTS -i ../_MCQ -m openai:o1-mini.json`
  * Uses the OpenAI model `o1-mini` to generate answers for MCQs in `MCQs.json` and stores results in the `_RESULTS` directory, in a file named `answers_openai:o1-mini.json`
* `python -m generate_answers -o ../_RESULTS -i MCQs.json -m "pb:argonne-private/AuroraGPT-IT-v4-0125`
  * Uses the Huggingface model `argonne-private/AuroraGPT-IT-v4-0125`, running on a Polaris compute node started via PBS, to generate answers for the same MCQs. Results are placed in `_RESULTS/answers_pb:argonne-private+AuroraGPT-IT-v4-0125.json`
 
Examples of running `score_answers.py`:
* `python -m score_answers -o _RESULTS -i MCQs.json -a openai:o1-mini.json -b openai:gpt-4o`
  * Uses the OpenAI model `gpt-4o` to score answers for MCQs in `MCQs.json` and stores results in `_RESULTS` directory, in a file named `answers_openai:o1-mini.json`
* `python -m score_answers -o _RESULTS -a pb:argonne-private/AuroraGPT-IT-v4-0125 -b openai:gpt-4o`
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
python -m lora_fine_tune -i <json-file> -o <model-directory>

# Full fine tune
python -m full_fine_tune -i <json-file> -o <model-directory>
```
Note:
* You need a file `hf_access_token.txt` if you want to publish models to HuggingFace.
* You need to edit the file to specify where to publish models in HuggingFace
* We are still debugging how to download and run published models

## Code for other useful things

Determine what models are currently running on ALCF inference service (see below for more info)
```
python -m check_alcf_service_status
```
Determine what answers have been generated and scored, and what additional runs could be performed, _given running models_, to generate and score additional answers. (You may want to submit runs to start models. Use `-m` flag to see what could be useful to submit.) 
```
python -m review_status -o <result-directory>
```
Perform runs of `generate_answers` and `grade_answers.py` to generate missing outputs. (See below for more info)
```
python -m run_missing_generates -o <result-directory>
```

### More on `check_alcf_service_status.py` 

The program `check_alcf_service_status.py` retrieves and processes status information from the
[ALCF Inference service](https://github.com/argonne-lcf/inference-endpoints),
and lists models currently running or queued to run. E.g., as follows, which shows six
models running and one queued. Models that are not accessed for some period are shut
down and queued models started. A request to a model that is not running adds it to the queue.
```
% python -m check_alcf_service_status
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
% python -m run_missing_generates -i 100-papers-qa.json -o output_files -a all -m 100 -s
python -m generate_and_grade_answers -i 100-papers-qa.json -o outputs -a 'Qwen/Qwen2-VL-72B-Instruct' -b 'gpt-4o' -c -q -s 0 -e 100
python -m generate_and_grade_answers -i 100-papers-qa.json -o outputs -a 'deepseek-ai/DeepSeek-V3' -b 'gpt-4o' -c -q -s 0 -e 100
python -m generate_and_grade_answers -i 100-papers-qa.json -o outputs -a 'mgoin/Nemotron-4-340B-Instruct-hf' -b 'gpt-4o' -c -q -s 0 -e 100
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




