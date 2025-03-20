#!/bin/zsh

# Default value for parameter p (number of threads for generate_mcqs.py,
# generate_answers.py, and score_answers.py)
p_value=8

# Parse command line options
while getopts "p:" opt; do
  case $opt in
    p)
      p_value="$OPTARG"
      ;;
    *)
      echo "Usage: $0 [-p value]"
      exit 1
      ;;
  esac
done

# Exit on any error
set -e

# Get models from config.yml using the dynamic value for -p
MODELS=("${(@f)$(python src/list_models.py -p "$p_value")}")

# Define aliases for up to 4 models (always at least two models)
ALIASES[1]="Model A"
ALIASES[2]="Model B"
ALIASES[3]="Model C"
ALIASES[4]="Model D"

# List the models at the start of the script
echo "Models to be used:"
if (( ${#MODELS[@]} > 0 )); then
  echo "  ${ALIASES[1]} (used for generating MCQs): ${MODELS[1]}"
  for (( i=2; i<=${#MODELS[@]}; i++ )); do
    echo "  ${ALIASES[i]}: ${MODELS[i]}"
  done
fi

echo "Step 1: Convert PDF to JSON."
python src/simple_parse.py

echo "Step 1: Generate MCQs (${ALIASES[1]})."
python src/generate_mcqs.py -p 1

echo "Step 2: Combine JSON files."
python src/combine_json_files.py -o MCQ-combined.json

echo "Step 3: Generate answers (all models)."
# Generate answers for each model using the dynamic value for -p
for (( i=1; i<=${#MODELS[@]}; i++ )); do
    echo "Begin generating answers with ${ALIASES[i]}..."
    python src/generate_answers.py -i MCQ-combined.json -m "${MODELS[i]}" -q -p "$p_value" &
done

wait

echo "Step 4: Score answers between all models."
# Score each model's answers using other models with the dynamic value for -p
for (( i=1; i<=${#MODELS[@]}; i++ )); do
    for (( j=1; j<=${#MODELS[@]}; j++ )); do
        if [ $i -ne $j ]; then
            echo "Begin scoring ${ALIASES[i]} answers using ${ALIASES[j]}..."
            python src/score_answers.py -a "${MODELS[i]}" -b "${MODELS[j]}" -q -p "$p_value" &
        fi
    done
done

wait

echo "Workflow completed."
