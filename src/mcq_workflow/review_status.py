#!/usr/bin/env python

"""
Review status of MCQ answer generation and scoring.
Identifies which models have generated answers and scores,
and what work remains to be done.
"""

import os
import glob
import json
import logging
import argparse
import subprocess
import sys
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yml"""
    try:
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"\n\nError: Failed to load config.yml: {e}")
        return None

def main():
    # Load config
    config = load_config()
    if not config:
        return 1
    parser = argparse.ArgumentParser(description='Review status of MCQ answer generation and scoring')
    parser.add_argument('-i', '--inputs', help=f'MCQ input directory (default: {config["directories"]["mcq"]} from config.yml)')
    parser.add_argument('-o', '--outputdir', help=f'Directory with results files (default: {config["directories"]["results"]} from config.yml)')
    parser.add_argument('-a', '--answer-models', help=f'Comma-separated list of models to check for answers (default: {config["model"]["name"]} from config.yml)')
    parser.add_argument('-s', '--score-models', help=f'Comma-separated list of models to check for scoring (default: {config["model_b"]["name"]} from config.yml)')
    parser.add_argument('-x', '--execute', help='Execute missing commands', action='store_true')
    parser.add_argument('-q', '--quiet', help='Reduce output verbosity', action='store_true')
    args = parser.parse_args()

    # Set up parameters
    # Use command-line args if provided, otherwise use config defaults
    input_dir = args.inputs if args.inputs else config['directories']['mcq']
    output_dir = args.outputdir if args.outputdir else config['directories']['results']
    execute = args.execute
    quiet = args.quiet

    # Set models to check
    answer_models = []
    if args.answer_models:
        answer_models = [m.strip() for m in args.answer_models.split(',')]
    else:
        # Use default from config if not specified on command line
        answer_models = [config['model']['name']]

    score_models = []
    if args.score_models:
        score_models = [m.strip() for m in args.score_models.split(',')]
    else:
        # Use default from config if not specified on command line
        score_models = [config['model_b']['name']]

    # Configure logging based on verbosity
    if quiet:
        logger.setLevel(logging.WARNING)
    # Validate directories
    if not os.path.exists(input_dir):
        logger.error(f"\n\nError: Input directory '{input_dir}' does not exist")
        return 1

    if not os.path.exists(output_dir):
        logger.error(f"\n\nError: Output directory '{output_dir}' does not exist")
        return 1
    # Check for MCQ input files
    mcq_files = glob.glob(f'{input_dir}/*.json') + glob.glob(f'{input_dir}/*.jsonl')
    if not mcq_files:
        logger.error(f"\n\nError: No MCQ files (*.json or *.jsonl) found in '{input_dir}'")
        return 1
    else:
        logger.info(f"Found {len(mcq_files)} MCQ input files in '{input_dir}'")

    # Find all existing answer files
    answer_files = glob.glob(f'{output_dir}/answers_*.jsonl')
    if answer_files:
        logger.info(f"Found {len(answer_files)} answer files in '{output_dir}'")
    else:
        logger.info(f"No answer files found in '{output_dir}'")
    
    # Extract model names from answer files
    answer_models_done = []
    for file in answer_files:
        model_name = os.path.basename(file).split('answers_')[1].split('.jsonl')[0].replace('+','/')
        answer_models_done.append(model_name)
    
    # Find all existing score files
    score_files = glob.glob(f'{output_dir}/scores_*.jsonl')
    if score_files:
        logger.info(f"Found {len(score_files)} score files in '{output_dir}'")
    else:
        logger.info(f"No score files found in '{output_dir}'")
    
    # Extract model pairs that have been scored
    scores_done = set()
    for file in score_files:
        # Parse the filename to extract both models
        basename = os.path.basename(file)
        if ':' in basename:
            parts = basename.split('scores_')[1].split('.jsonl')[0].split(':')
            if len(parts) == 2:
                model_a = parts[0].replace('+', '/')
                model_b = parts[1].replace('+', '/')
                scores_done.add((model_a, model_b))
    
    # If no generated answers found for the specified models, inform the user
    if not any(model in answer_models_done for model in answer_models):
        logger.info(f"No generated answers found for specified models: {', '.join(answer_models)}")
    
    # If no models were discovered in the output directory
    if not answer_models_done:
        logger.info(f"No answer files found for any models in '{output_dir}'")
    
    # Determine what commands need to be run
    answer_commands = []
    for model in answer_models:
        if model not in answer_models_done:
            cmd = f'python generate_answers.py -i {input_dir} -o {output_dir} -m {model}'
            answer_commands.append(cmd)
    
    score_commands = []
    for model_a in answer_models_done:
        for model_b in score_models:
            if (model_a, model_b) not in scores_done:
                cmd = f'python score_answers.py -o {output_dir} -a {model_a} -b {model_b}'
                score_commands.append(cmd)
    
    # Report status
    if not quiet:
        if not answer_models and not score_models:
            logger.error("\n\nError: No models to check and no existing models found")
            logger.info("Use -a and -s options to specify models to check, for example:")
            logger.info("  -a \"model1,model2\" -s \"model3,model4\"")
            return 1

        total_work = len(answer_commands) + len(score_commands)
        if total_work > 0:
            logger.info(f"\n===== Current Status ({total_work} tasks remaining) =====")
        else:
            logger.info("\n===== Current Status (all tasks complete) =====")
        
        logger.info("\nAnswers Generated:")
        for model in sorted(answer_models_done):
            symbol = "✓" if model in answer_models else " "
            logger.info(f"  {symbol} {model}")
        
        missing_answers = set(answer_models) - set(answer_models_done)
        if missing_answers:
            logger.info("\nAnswers Missing:")
            for model in sorted(missing_answers):
                logger.info(f"  ✗ {model}")
        
        # Display scoring status
        logger.info("\nScoring Status:")
        total_pairs = len(answer_models_done) * len(score_models)
        completed_pairs = len(scores_done)
        logger.info(f"  Completed: {completed_pairs} of {total_pairs} possible model pairs")
        
        if scores_done:
            logger.info("\nScores Complete:")
            for model_a, model_b in sorted(scores_done):
                logger.info(f"  ✓ {model_a} scored by {model_b}")
        
        # Display missing score pairs
        missing_scores = []
        for model_a in answer_models_done:
            for model_b in score_models:
                if (model_a, model_b) not in scores_done:
                    missing_scores.append((model_a, model_b))
        
        if missing_scores:
            logger.info("\nScores Missing:")
            for model_a, model_b in sorted(missing_scores):
                logger.info(f"  ✗ {model_a} needs scoring by {model_b}")
        
        # Display commands to run
        if answer_commands or score_commands:
            logger.info("\n===== Commands to Run =====")
            
            if answer_commands:
                logger.info("\nCommands to generate answers:")
                for cmd in answer_commands:
                    logger.info(f"  {cmd}")
            
            if score_commands:
                logger.info("\nCommands to generate scores:")
                for cmd in score_commands:
                    logger.info(f"  {cmd}")
    
    # Execute commands if requested
    if execute and (answer_commands or score_commands):
        logger.info("\n===== Executing Commands =====")
        
        # Execute answer generation commands
        if answer_commands:
            logger.info("\nGenerating answers...")
            for cmd in answer_commands:
                logger.info(f"\nExecuting: {cmd}")
                try:
                    subprocess.run(cmd, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Command failed: {e}")
        
        # Execute scoring commands
        if score_commands:
            logger.info("\nGenerating scores...")
            for cmd in score_commands:
                logger.info(f"\nExecuting: {cmd}")
                try:
                    subprocess.run(cmd, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Command failed: {e}")
    
    return 0
if __name__ == "__main__":
    sys.exit(main())
