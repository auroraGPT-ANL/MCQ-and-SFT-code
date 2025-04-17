#!/usr/bin/env python

import sys
import json
import logging
import os
import argparse
from common.model_access import Model
from common.config import logger

class TestResults:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.results = []
        self.file_logger = self._setup_file_logger()
        
    def _setup_file_logger(self):
        # Set up file logging (always detailed)
        handler = logging.FileHandler('test_model.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger = logging.getLogger("test_model_verification")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        return logger
    
    def log(self, message, level=logging.INFO):
        # Always log to file
        self.file_logger.log(level, message)
        # Print to console if verbose
        if self.verbose:
            print(message)
            
    def add_result(self, message):
        self.results.append(message)
        self.log(message)
        if not self.verbose:
            print(message)

def test_model_run(results, model, prompt_type, prompt_text, expected_format=None):
    """
    Run a test using the specified model and prompt.
    """
    results.log(f"Testing {prompt_type}...")
    response = model.run(prompt_text)
    results.log(f"Response received: {response[:100]}...")
    
    if not response:
        results.log(f"No response received for {prompt_type}", logging.ERROR)
        return False, response
    
    if expected_format == 'json':
        try:
            json_data = json.loads(response)
            results.log(f"Successfully parsed JSON response for {prompt_type}")
            return True, json_data
        except json.JSONDecodeError:
            results.log(f"Failed to parse JSON response for {prompt_type}", logging.ERROR)
            return False, response
    
    return True, response

def test_augmented_chunk_generation(results, model):
    """Test augmented chunk generation"""
    prompt = "Given the following chunk of text, please summarize the key points and augment with relevant details: 'The test model implementation helps with offline testing.'"
    success, response = test_model_run(results, model, "augmented chunk generation", prompt, 'json')
    
    if success and isinstance(response, dict) and "augmented_text" in response:
        results.add_result("✅ Augmented chunk generation test passed")
        return True
    else:
        results.add_result("❌ Augmented chunk generation test failed")
        return False

def test_mcq_generation(results, model):
    """Test MCQ generation"""
    prompt = "Below is some content called augmented_chunk. Please generate one multiple-choice question based on this content with four options (A, B, C, D). Exactly one option should be correct. Provide the correct answer as well.\n\naugmented_chunk: The test model in this codebase provides pre-defined responses for testing."
    success, response = test_model_run(results, model, "MCQ generation", prompt, 'json')
    
    if success and isinstance(response, dict) and all(k in response for k in ("question", "options", "answer")):
        results.add_result("✅ MCQ generation test passed")
        return True
    else:
        results.add_result("❌ MCQ generation test failed")
        return False

def test_verification(results, model):
    """Test verification response"""
    prompt = 'Please verify this MCQ and provide a score from 1-10:\naugmented_chunk: "The test model provides responses for testing."\nMCQ: {"question": "What is the purpose of the test model?", "options": "A. Option 1\nB. Option 2\nC. Option 3\nD. Option 4", "answer": "B"}'
    success, response = test_model_run(results, model, "verification", prompt, 'json')
    
    if success and isinstance(response, dict) and all(k in response for k in ("question", "answer", "score")):
        results.add_result("✅ Verification test passed")
        return True
    else:
        results.add_result("❌ Verification test failed")
        return False

def test_scoring(results, model):
    """Test scoring response"""
    prompt = "How consistent is the user answer with the reference answer? User answer: 'Testing'. Reference answer: 'The test model provides testing capabilities.'"
    success, response = test_model_run(results, model, "scoring", prompt, 'string')
    
    if success:
        try:
            score = float(response.strip())
            results.add_result(f"✅ Scoring test passed with score: {score}")
            return True
        except ValueError:
            results.add_result(f"❌ Scoring test failed - response not numeric: {response}")
            return False
    else:
        results.add_result("❌ Scoring test failed")
        return False

def run_all_tests(verbose=False):
    """Run all tests with different model variants"""
    results = TestResults(verbose)
    test_stats = {
        "default": {"passed": 0, "total": 0},
        "mcq": {"passed": 0, "total": 0},
        "score": {"passed": 0, "total": 0}
    }
    
    # Test with default 'all' variant
    if verbose:
        print("=" * 50)
        print("TESTING DEFAULT (ALL) VARIANT")
        print("=" * 50)
    model_all = Model("test:all")
    
    tests = [
        ("Augmented Chunk Generation", test_augmented_chunk_generation),
        ("MCQ Generation", test_mcq_generation),
        ("Verification", test_verification),
        ("Scoring", test_scoring)
    ]
    
    for test_name, test_func in tests:
        test_stats["default"]["total"] += 1
        if test_func(results, model_all):
            test_stats["default"]["passed"] += 1
    
    # Test with MCQ variant
    if verbose:
        print("\n" + "=" * 50)
        print("TESTING MCQ VARIANT")
        print("=" * 50)
    model_mcq = Model("test:mcq")
    
    test_stats["mcq"]["total"] += 1
    if test_mcq_generation(results, model_mcq):
        test_stats["mcq"]["passed"] += 1
    
    # Test with score variant
    if verbose:
        print("\n" + "=" * 50)
        print("TESTING SCORE VARIANT")
        print("=" * 50)
    model_score = Model("test:score")
    
    test_stats["score"]["total"] += 1
    if test_scoring(results, model_score):
        test_stats["score"]["passed"] += 1
    
    # Print summary
    print("\nTEST SUMMARY")
    if verbose:
        print("=" * 50)
    
    for variant, stats in test_stats.items():
        print(f"{variant.upper()} variant: {stats['passed']}/{stats['total']} tests passed")
    
    total_passed = sum(stats["passed"] for stats in test_stats.values())
    total_tests = sum(stats["total"] for stats in test_stats.values())
    print(f"OVERALL: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\nAll tests passed successfully! The automated content assessment workflow is functioning correctly:")
        print("✓ Content augmentation")
        print("✓ MCQ generation")
        print("✓ Answer verification")
        print("✓ Response scoring")
    else:
        print("\nSome workflow components failed their tests. Check test_model.log for detailed error information.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Verify the automated content assessment workflow components'
    )
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output (default: concise output)')
    args = parser.parse_args()
    
    success = run_all_tests(args.verbose)
    sys.exit(0 if success else 1)
