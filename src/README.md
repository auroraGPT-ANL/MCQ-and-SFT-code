# Source Code Documentation

## Main Workflow Components

The `run_workflow.sh` script orchestrates the main workflow, which consists of several Python scripts executed in sequence:

1. `simple_parse.py`: Converts PDF files into JSON format for processing
2. `generate_mcqs.py`: Generates Multiple Choice Questions (MCQs) using the specified model
3. `combine_json_files.py`: Combines individual JSON files containing MCQs into a single file
4. `generate_answers.py`: Uses specified models to generate answers for the MCQs
5. `score_answers.py`: Evaluates the quality of generated answers using other models

### Workflow Parameters

The `run_workflow.sh` script accepts the following options:
- `-p <value>`: Number of parallel threads (default: 8)
- `-v`: Enable verbose output
- `-n <number>`: Select a random subset of MCQs

## Test Infrastructure

### Test Components

1. `test_model.py`: Implements a test model that provides predefined responses for offline testing
2. `test_model_verification.py`: Comprehensive test suite for verifying the test model implementation

### Running Tests

To run the test model verification suite:

```bash
python src/test_model_verification.py
```

To test the workflow with a specific input file:

```bash
./src/test_workflow.sh -i input.pdf
```

To test the workflow with verbose output:

```bash
./src/test_workflow.sh -i input.pdf -v
```

### Test Coverage

The test suite verifies:
1. Augmented chunk generation
2. MCQ generation
3. Answer verification
4. Response scoring

Each test component produces detailed logs in `test_model.log` that can be used for debugging and verification.

## Development Tools

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
