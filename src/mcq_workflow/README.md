# Source Code Documentation

### Note this needs to be updated with the reorg of the repo, where modules are used
Thus one should be able to run the python scripts such as *foo_bar.py* using
```bash
python -m foo_bar
```
(note omitting the *.py*)


## Main Workflow Components

The `run_workflow.sh` script orchestrates the main workflow, which comprises several Python scripts executed in sequence:

1. `simple_parse.py`: Converts PDF files into JSON format for processing
2. `generate_mcqs.py`: Processes input text in parallel chunks to generate Multiple Choice Questions (MCQs)
3. `combine_json_files.py`: Combines individual JSON files containing MCQs into a single file
4. `generate_answers.py`: Uses specified models to generate answers for the MCQs
5. `score_answers.py`: Evaluates the quality of generated answers using other models

### Workflow Parameters

The `run_workflow.sh` script accepts the following options:
- `-p <value>`: Number of parallel threads (default: 8)
- `-v`: Enable verbose output
- `-n <number>`: Select a random subset of MCQs


## Test Infrastructure

The testing infrastructure consists of two main components that work together to enable both offline
testing and workflow verification.

### 1. Workflow Integration Test (`test_workflow.sh`)

Integration testing focused on the complete system:
- Tests the entire workflow pipeline, making sure all scripts work together correctly
- Uses a simplified setup with just two models (A and B in config.yml)
- Verifies integration between all components
- Uses real PDF input for testing
- Tests the complete workflow:
  * PDF to JSON conversion
  * MCQ generation
  * Answer generation
  * Score computation
- Creates and processes temporary files (does not affect user data in \_PAPERS and other working directories)

To run the workflow test with a specific input:
```bash
./src/test_workflow.sh -i input.pdf [-v]
```

### 2. Test Model Verification (`test_model_verification.py`)

To facilitate offline testing without delays associated with model interactions, a stub model ('test\_model.py') is
used. It provides predefined ressponses to test workflow components without requiring access to models.
This code ensures that the test model functions properly with each of the workflow components.

Unit testing focused on validating the stub model implementation:
- Tests the offline test model implementation (src/test\_model.py)
- Verifies that the stub model provides appropriate responses
- Tests three specific variants (all, mcq, score)
- Uses predefined test prompts
- Validates all required operations:
  * Augmented chunk generation
  * MCQ creation
  * Question verification
  * Answer scoring
- Outputs detailed logs to `test_model.log`

To run the test model verification:
```bash
python src/test_model_verification.py
```
Use the '-v' or '--verbose' option for more detailed test messages.


## Other Tools

For developers working on extending the codebase:

To extract Q&A pairs from JSON:
```bash
python src/extract_qa.py -i input.json -o output.json
```

To review the processing status of models:
```bash
python src/review_status.py -o results_directory
```

To run missing generates for specific models:
```bash
python src/run_missing_generates.py -i input.json -o output_dir -a model_name
```
