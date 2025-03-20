#!/bin/zsh

# Exit on any error
set -e

echo "Step 1: Convert PDF to JSON."
python src/simple_parse.py

echo "Step 1: Generate MCQs (Model A)."
python src/generate_mcqs.py -p 1

echo "Step 2: Combine JSON files."
python src/combine_json_files.py -o MCQ-combined.json 

echo "Step 3: Generate answers (all models)."
# Get models from config.yml
MODELS=("${(@f)$(python src/list_models.py -p 8)}")

# Generate answers for each model
for MODEL in "${MODELS[@]}"; do
    echo "Generating answers with $MODEL..."
    python src/generate_answers.py -i MCQ-combined.json -m "$MODEL" -p 8
done

echo "Step 4: Score answers between all models."
# Score each model's answers using other models
for MODEL_A in "${MODELS[@]}"; do
    for MODEL_B in "${MODELS[@]}"; do
        if [ "$MODEL_A" != "$MODEL_B" ]; then
            echo "Scoring $MODEL_A answers using $MODEL_B..."
            python src/score_answers.py -a "$MODEL_A" -b "$MODEL_B" -p 8
        fi
    done
done

echo "Workflow completed."

