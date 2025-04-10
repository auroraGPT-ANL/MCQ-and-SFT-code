#!/bin/zsh

# Default value for parameter p (number of threads for generate_mcqs.py, generate_answers.py, and score_answers.py)
p_value=8
v_flag=""
n_value=""

# Parse command line options
while getopts "p:vn:" opt; do
  case $opt in
    p)
      p_value="$OPTARG"
      ;;
    v)
      v_flag="-v"
      ;;
    n)
      n_value="$OPTARG"
      ;;
    *)
      echo "Usage: $0 [-p value] [-v] [-n number]"
      exit 1
      ;;
  esac
done

# Exit on any error
set -e

# List models from config.yml using the dynamic value for -p
MODELS=("${(@f)$(python src/list_models.py -p "$p_value")}")

# Define aliases for up to 4 models (always at least two models)
ALIASES[1]="Model A"
ALIASES[2]="Model B"
ALIASES[3]="Model C"
ALIASES[4]="Model D"

# List the models at the start of the script
echo "Models to be used:"
if (( ${#MODELS[@]} > 0 )); then
  echo "  ${ALIASES[1]}: ${MODELS[1]} (used for generating MCQs)"
  for (( i=2; i<=${#MODELS[@]}; i++ )); do
    echo "  ${ALIASES[i]}: ${MODELS[i]}"
  done
fi

echo "Step 1: Convert PDF to JSON."
python src/simple_parse.py

echo "Step 1: Generate MCQs (${ALIASES[1]})."
python src/generate_mcqs.py -p "$p_value" $v_flag

echo "Step 2: Combine JSON files."
python src/combine_json_files.py -o MCQ-combined.json

# If -n is specified, select a subset of MCQs at random
if [ -n "$n_value" ]; then
    echo "Selecting $n_value MCQs at random..."
    python src/select_mcqs_at_random.py -i MCQ-combined.json -o MCQ-subset.json -n "$n_value"
    input_file="MCQ-subset.json"
else
    input_file="MCQ-combined.json"
fi

echo "Step 3: Generate answers (all models)."
# Build a single string with the actual model names
model_names=$(printf "%s, " "${MODELS[@]}")
model_names=${model_names%, }  # Remove trailing comma and space
echo "Generating answers with models: ${model_names}"

# Generate answers for each model using the dynamic value for -p
for (( i=1; i<=${#MODELS[@]}; i++ )); do
    python src/generate_answers.py -i "$input_file" -m "${MODELS[i]}" -q -p "$p_value" $v_flag &
done

wait

echo "Step 4: Score answers between all models."
# Score each model's answers using other models with the dynamic value for -p
for (( i=1; i<=${#MODELS[@]}; i++ )); do
    for (( j=1; j<=${#MODELS[@]}; j++ )); do
        if [ $i -ne $j ]; then
            echo "Begin scoring ${ALIASES[i]} answers using ${ALIASES[j]}..."
            python src/score_answers.py -a "${MODELS[i]}" -b "${MODELS[j]}" -q -p "$p_value" $v_flag &
        fi
    done
done

wait

echo "Workflow completed."

