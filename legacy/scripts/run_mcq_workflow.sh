#!/usr/bin/env zsh
#set -e

# Handle CLI options when script is sourced
if [[ $ZSH_EVAL_CONTEXT =~ :file$ ]]; then
  # Script is being sourced
  if [[ $# -gt 0 ]]; then
    # Set positional parameters to the arguments passed to source
    set -- "$@"
  fi
fi

# Start timer
start_time=$(date +%s)

# Determine the project root (one level above this script)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
PROJECT_ROOT=$(dirname "$PROJECT_ROOT")

#echo SCRIPT DIR $SCRIPT_DIR
#echo PROJECT ROOT $PROJECT_ROOT

# Ensure secrets.yml exists
if [ ! -f "$PROJECT_ROOT/secrets.yml" ]; then
  echo "Please create secrets.yml. If using argo models, populate with:"
  cat << 'EOF'
argo:
    username: YOUR_ARGO_USERNAME
EOF
  return
fi

# Temporarily drop into the repo root directory
cd "$PROJECT_ROOT"

# Ensure src/ is on PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Default value for parameter p (number of threads for generate_mcqs.py, generate_answers.py, and score_answers.py)

# Reset OPTIND to ensure proper option processing when script is sourced
OPTIND=1

# Initialize variables
p_value=8
v_flag=""
n_value=""

# Parse command line options
while getopts "p:vn:" opt; do
  case $opt in
    p) p_value="$OPTARG" ;;
    v) v_flag="-v" ;;
    n) n_value="$OPTARG" ;;
    *) echo "Usage: $0 [-p value] [-v] [-n number]" && return ;;
  esac
done

# List models from config.yml using the dynamic value for -p
MODELS=("${(@f)$(python -m common.list_models -p "$p_value")}")

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

# Display active options
echo "Options:"
echo "  -p $p_value"
if [ -n "$v_flag" ]; then
  echo "  -v"
fi
if [ -n "$n_value" ]; then
  echo "  -n $n_value"
fi

echo "Step 1: Convert PDF to JSON ($(date))."
python -m common.simple_parse

echo "Step 1: Generate MCQs (${ALIASES[1]}) ($(date))."
python -m mcq_workflow.generate_mcqs -p "$p_value" $v_flag

echo "Step 2: Combine JSON files ($(date))."
python -m common.combine_json_files -o MCQ-combined.json

# If -n is specified, select a subset of MCQs at random
if [ -n "$n_value" ]; then
  echo "Selecting $n_value MCQs at random..."
  python -m mcq_workflow.select_mcqs_at_random -i MCQ-combined.jsonl -o MCQ-subset.jsonl -n "$n_value"
  input_file="MCQ-subset.json"
else
  input_file="MCQ-combined.json"
fi

echo "Step 3: Generate answers (all models) ($(date))."
# Build a single string with the actual model names
model_names=$(printf "%s, " "${MODELS[@]}")
model_names=${model_names%, }  # Remove trailing comma and space
echo "Generating answers with models: ${model_names}"

# Generate answers for each model using the dynamic value for -p
for (( i=1; i<=${#MODELS[@]}; i++ )); do
  python -m mcq_workflow.generate_answers -i "$input_file" -m "${MODELS[i]}" -p "$p_value" $v_flag &
done
wait

echo "Step 4: Score answers between all models ($(date))."
# Score each model's answers using other models with the dynamic value for -p
for (( i=1; i<=${#MODELS[@]}; i++ )); do
  for (( j=1; j<=${#MODELS[@]}; j++ )); do
    if [ $i -ne $j ]; then
      echo "Begin scoring ${ALIASES[i]} answers using ${ALIASES[j]}..."
      python -m mcq_workflow.score_answers -a "${MODELS[i]}" -b "${MODELS[j]}" -p "$p_value" $v_flag &
    fi
  done
done
wait

echo "Workflow completed ($(date))."

# End timer and print elapsed time
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
# Format as HH:MM:SS
printf "Total elapsed time: %02d:%02d:%02d (hh:mm:ss)\n" \
  $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))

