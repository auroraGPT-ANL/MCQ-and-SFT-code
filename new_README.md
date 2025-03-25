# Code for Creating and Scoring Multiple Choice Questions (MCQs) from Papers

This repository contains Python programs for:
* Generating and evaluating Multiple Choice Questions (MCQs) from scientific papers
* Fine-tuning models based on supplied data
* Other useful tools for working with AI models

Please email {foster|stevens|catlett}@anl.gov if you see things that are unclear or missing.

## Workflow Overview

This pipeline converts scientific papers in **PDF format** into JSON and then uses AI models of your choice to generate **multiple-choice questions (MCQs)**, **answers**, and **scores** of those answers.

### Workflow Steps

[**(flowchart)**](https://github.com/auroraGPT-ANL/MCQ-and-SFT-code/blob/CeC/MCQ-Workflow.png)

1. Convert PDFs (papers) to JSON representations
2. Generate MCQs from JSON representations
3. Combine multiple MCQ JSON files into a single file
4. Select a subset of MCQs (optional)
5. Generate additional answers for MCQs (using a different model than used to generate the initial MCQs)
6. Score AI-generated answers using another AI model
7. Review the status of MCQ generation and scoring

## Setup Instructions

### Set up to access ALCF Inference Service

**Before you start:** We recommend you follow the instructions for 
[ALCF Inference Service Prerequisites](https://github.com/argonne-lcf/inference-endpoints?tab=readme-ov-file#%EF%B8%8F-prerequisites)
to set up your ALCF auth token, required to access models via the inference service.
(You need to download and run `inference_auth_token.py`.)

### Clone this repository

```bash
git clone git@github.com:auroraGPT-ANL/MCQ-and-SFT-code.git
cd MCQ-and-SFT-code
```

### Set Up Your Working Directory

Ensure your working directory has subdirectories for storing input and output files. The names
of files and folders don't matter, but these are the names specified in config.yml. If you want
to place data elsewhere, update the directories section in `config.yml`.

```bash
mkdir _PAPERS _JSON _MCQ _RESULTS
```

- `_PAPERS/`  → **original PDF papers**
- `_JSON/`    → **parsed text in JSON format**
- `_MCQ/`     → **generated MCQs in JSON format**
- `_RESULTS/` → **AI-generated answers and scores**

Put your papers (in PDF form) in **_PAPERS**.

### Set Up and Activate Your Conda Environment

If you already have a Conda environment you want to keep using, update it with 
any missing dependencies needed for this workflow:
```bash
conda env update --name <your_conda_env> --file environment.yml
```

Otherwise, create a new Conda environment:
```bash
conda env create -f environment.yml
conda activate globus_env
```
(**Note:** If you get `CondaValueError: prefix already exists`, edit `environment.yml` and change the `name:`,
then create and activate that env.)

### Configure Your Models in config.yml

Edit *config.yml* to specify the models you want to run. Note you need to specify the location and the model name. This code has been tested with alcf:(*model_name*) and local.

Specify 2-4 models. The first (Model A) will be used to generate MCQs. All models will then be used to generate answers and to score one another's answers.

In *config.yml* you can also set the default for parallelization (default is 4-way) as most of the scripts here fire up multiple threads to speed things up.

## Two Paths to Run the Workflow

### Quick Path: Run the Entire Workflow at Once

You can run the entire workflow by invoking the shell script:

```bash
./src/run_workflow.sh [options]
```

The available command-line options are:

* **-p \<value\>**  
  Specifies the number of threads or parallel processes to use for MCQ generation,
  answer generation, and scoring. Default is 8. For example, to run with
  12-way parallelism:
  ```bash
  ./src/run_workflow.sh -p 12
  ```

* **-v**  
  Enables verbose mode, which may display additional logging or debugging
  messages during the execution of various Python scripts.

* **-n \<integer\>**  
  Specifies how many MCQs to randomly select after combining the
  MCQ output files. If set, the script will create a smaller subset
  (named MCQ-subset.json) containing \<integer\> random MCQs and then use 
  that subset for answer generation. If omitted, **all** MCQs in MCQ-combined.json will be used.

Example Commands:

Run the entire workflow with 8 threads (the default) and verbose mode
```bash
./src/run_workflow.sh -v
```

Run the entire workflow with 12 threads, selecting 20 random MCQs
```bash
./src/run_workflow.sh -p 12 -n 20
```

Run the entire workflow with 4 threads, using all MCQs (no subset), no verbose logging
```bash
./src/run_workflow.sh -p 4
```

### Detailed Step-by-Step Instructions

#### 1. Convert PDFs to JSON
Extract text from PDFs using a simple parser:
```bash
python src/simple_parse.py 
```
**Note:** You can specify input and output with, e.g., `-i _PAPERS -o _JSON`, otherwise the
code will default to the directories specified in `config.yml`

Alternatively, you can use **AdaParse** (higher-quality parser, still in testing). 
[More details](https://github.com/7shoe/AdaParse/tree/main)

#### 2. Generate MCQs Using an AI Model
To generate MCQs from parsed JSON files:

1. **Authenticate with ALCF inference service (if not already done):**
   ```bash
   python src/inference_auth_token.py authenticate
   ```
   
2. **(Optional) Check which models are running**
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
   ```bash
   python src/inference_auth_token.py authenticate --force
   ```
   
3. **Run MCQ generation:**
   This step uses generate_mcqs.py to divide text into chunks, generate MCQs, and
   include reference answers.

   If you have set up your models in *config.yml*:
   ```bash
   python src/generate_mcqs.py 
   ```

   If you want to specify your model in the command line:
   ```bash
   python src/generate_mcqs.py -m 'alcf:mistralai/Mistral-7B-Instruct-v0.3'
   ```
   
   By default, the code displays a progress bar. You can display informational
   progress messages using the *-v / --verbose* option or you can suppress all
   information and progress bar using *-q / --quiet*.

#### 3. Combine multiple MCQ JSON files into a single file
```bash
python src/combine_json_files.py -o MCQ-combined.json
```

Input for this step is taken from the directory specified in *config.yml*.
Here you can override by specifying -i (--input), but you must specify
the filename for your combined file (-o or --output) as shown here.

#### 4. Select a Random Subset of MCQs for Further Processing (optional)
If you want to randomly select a subset of MCQs from the generated JSON files, use 
`select_mcqs_at_random.py`, specifying the number of MCQs to select.
For example, to select 17 MCQs:
```bash
python src/select_mcqs_at_random.py -i MCQ-combined.json -o MCQ-subset.json -n 17
```
You must specify the filenames for your combined and subset files as shown here.

#### 5. Generate Answers for MCQs Using a Different Model
This step uses the model specified in *config.yml* to generate **new answers** for
the selected MCQs. If you want to continue using the model in *config.yml* for this 
step:
```bash
python src/generate_answers.py -i MCQ-subset.json
```

You can specify a different model here using the -m (--model) option
if you don't want to use the model specified in config.yml. E.g.,:

```bash
python src/generate_answers.py -i MCQ-subset.json \
       -m 'alcf:meta-llama/Meta-Llama-3-70B-Instruct'
```
Note the input shown here is `MCQ-subset.json` which assumes that you
performed step 4; otherwise use `MCQ-combined.json` 
(or whatever filename you used for output in step 3)

#### 6. Score AI-Generated Answers
An AI model evaluates and scores the generated answers against reference answers.
By default this script assumes that the model specified in *config.yml* was used
by generate_answers.py (in the previous step) and will use the model specified
as *model_b* in *config.yml*:
```bash
python src/score_answers.py
```

You can override *config.yml* settings to specify models in the command line, e.g.,:
```bash
python src/score_answers.py \
       -a 'alcf:meta-llama/Meta-Llama-3-70B-Instruct' \
       -b 'alcf:mistralai/Mistral-7B-Instruct-v0.3'
```
Input and output directories default to the directories
specified in config.yml but can
be overriden with -i (--input) and/or -o (--output)  on the command line. 
- **Input:**  `_RESULTS/answers_<model-A>.json`
- **Output:** `_RESULTS/scores_<locn-A>:<model-A>_<locn-B>:<model-B>.json`
- **Note:** Any `/` in model names is replaced with `+` in filenames.

#### 7. Review MCQ Generation and Scoring Status
To check progress and see which MCQs are answered/scored:
```bash
python src/review_status.py -i MCQ-combined.json 
```
- This script identifies missing or incomplete processing steps.
- As earlier, output defaults to the directory specified in config.yml 
  (`_RESULTS`) but can be overriden on the coammand line with -o *directory-name*.

---

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

Determine what models are currently running on ALCF inference

