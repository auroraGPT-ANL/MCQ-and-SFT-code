#!/bin/zsh

# For testing we force multithreading to use 2 threads.
p_value=2
v_flag=""
n_value=4
input_pdf=""

# Parse arguments: -i|--input requires a PDF file; -v for verbose.
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -i|--input)
      if [[ -n $2 ]]; then
        input_pdf="$2"
        shift 2
      else
        echo "Error: -i|--input requires a PDF file as argument."
        exit 1
      fi
      ;;
    -v)
      v_flag="-v"
      shift
      ;;
    *)
      echo "Usage: $0 -i|--input <pdf file> [-v]"
      exit 1
      ;;
  esac
done

# Ensure an input PDF file was provided.
if [ -z "$input_pdf" ]; then
    echo "Error: Input PDF file is required. Usage: $0 -i|--input <pdf file> [-v]"
    exit 1
fi

# If the input is a file (and not a directory), create a temporary directory and copy the file.
if [ -f "$input_pdf" ]; then
    temp_dir=$(mktemp -d)
    cp "$input_pdf" "$temp_dir"
    input_dir="$temp_dir"
    echo "Using temporary directory $temp_dir for processing the input file."
else
    input_dir="$input_pdf"
fi

# Exit on any error
set -e

# For testing, we only use the first two models (Model A and Model B).
# List models from config.yml using p_value=2.
MODELS=("${(@f)$(python src/list_models.py -p "$p_value")}")
# Restrict to just the first two models.
MODELS=( "${MODELS[1]}" "${MODELS[2]}" )

# Define fixed aliases for the two models.
ALIASES=("Model A" "Model B")

echo "Models to be used:"
echo "  ${ALIASES[1]}: ${MODELS[1]} (used for generating MCQs)"
echo "  ${ALIASES[2]}: ${MODELS[2]}"

echo "Step 1: Convert PDF to JSON from input: $input_pdf"
python src/simple_parse.py -i "$input_dir"

echo "Step 1: Generate MCQs (${ALIASES[1]})."
python src/generate_mcqs.py -p "$p_value" $v_flag

echo "Step 2: Combine JSON files."
python src/combine_json_files.py -o MCQ-combined.json

# For test workflow, we use the fixed combined JSON file.
input_file="MCQ-combined.json"

echo "Step 3: Generate answers (using ${ALIASES[1]} and ${ALIASES[2]})."
# Generate answers for Model A and Model B.
for i in 1 2; do
    echo "Generating answers with ${ALIASES[i]}..."
    python src/generate_answers.py -i "$input_file" -m "${MODELS[i]}" -q -p "$p_value" $v_flag &
done

wait

echo "Step 4: Score answers between ${ALIASES[1]} and ${ALIASES[2]}."
echo "Scoring ${ALIASES[1]} answers with ${ALIASES[2]}..."
python src/score_answers.py -a "${MODELS[1]}" -b "${MODELS[2]}" -q -p "$p_value" $v_flag &
echo "Scoring ${ALIASES[2]} answers with ${ALIASES[1]}..."
python src/score_answers.py -a "${MODELS[2]}" -b "${MODELS[1]}" -q -p "$p_value" $v_flag &

wait

echo "Test workflow completed."

# Cleanup: Remove temporary directory if it was created.
if [ -n "$temp_dir" ]; then
    rm -rf "$temp_dir"
    echo "Temporary directory $temp_dir removed."
fi

