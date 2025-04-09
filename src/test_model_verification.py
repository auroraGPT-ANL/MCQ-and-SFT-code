#!/usr/bin/env python

import sys
import json
import logging
import os
from model_access import Model
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_model.log')
    ]
)

logger = logging.getLogger("test_model_verification")
config.logger = logger

def test_model_run(model, prompt_type, prompt_text, expected_format=None):
    """
    Run a test using the specified model and prompt.
    
    Args:
        model: The Model instance to test
        prompt_type: Description of the test type (for logging)
        prompt_text: The prompt to send to the model
        expected_format: Optional format check ('json' or 'string')
        
    Returns:
        A tuple of (success, response) where success is a boolean
    """
    logger.info(f"Testing {prompt_type}...")
    response = model.run(prompt_text)
    logger.info(f"Response received: {response[:100]}...")
    
    if not response:
        logger.error(f"No response received for {prompt_type}")
        return False, response
    
    if expected_format == 'json':
        try:
            json_data = json.loads(response)
            logger.info(f"Successfully parsed JSON response for {prompt_type}")
            return True, json_data
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response for {prompt_type}")
            return False, response
    
    return True, response

def test_augmented_chunk_generation(model):
    """Test augmented chunk generation"""
    prompt = "Given the following chunk of text, please summarize the key points and augment with relevant details: 'The test model implementation helps with offline testing.'"
    success, response = test_model_run(model, "augmented chunk generation", prompt, 'json')
    
    if success and isinstance(response, dict) and "augmented_text" in response:
        logger.info("✅ Augmented chunk generation test passed")
        return True
    else:
        logger.error("❌ Augmented chunk generation test failed")
        return False

def test_mcq_generation(model):
    """Test MCQ generation"""
    prompt = "Below is some content called augmented_chunk. Please generate one multiple-choice question based on this content with four options (A, B, C, D). Exactly one option should be correct. Provide the correct answer as well.\n\naugmented_chunk: The test model in this codebase provides pre-defined responses for testing."
    success, response = test_model_run(model, "MCQ generation", prompt, 'json')
    
    if success and isinstance(response, dict) and all(k in response for k in ("question", "options", "answer")):
        logger.info("✅ MCQ generation test passed")
        return True
    else:
        logger.error("❌ MCQ generation test failed")
        return False

def test_verification(model):
    """Test verification response"""
    prompt = 'Please verify this MCQ and provide a score from 1-10:\naugmented_chunk: "The test model provides responses for testing."\nMCQ: {"question": "What is the purpose of the test model?", "options": "A. Option 1\nB. Option 2\nC. Option 3\nD. Option 4", "answer": "B"}'
    success, response = test_model_run(model, "verification", prompt, 'json')
    
    if success and isinstance(response, dict) and all(k in response for k in ("question", "answer", "score")):
        logger.info("✅ Verification test passed")
        return True
    else:
        logger.error("❌ Verification test failed")
        return False

def test_scoring(model):
    """Test scoring response"""
    prompt = "How consistent is the user answer with the reference answer? User answer: 'Testing'. Reference answer: 'The test model provides testing capabilities.'"
    success, response = test_model_run(model, "scoring", prompt, 'string')
    
    if success:
        try:
            # Check if response can be converted to an integer or float
            score = float(response.strip())
            logger.info(f"✅ Scoring test passed with score: {score}")
            return True
        except ValueError:
            logger.error(f"❌ Scoring test failed - response not numeric: {response}")
            return False
    else:
        logger.error("❌ Scoring test failed")
        return False

def run_all_tests():
    """Run all tests with different model variants"""
    results = {
        "default": {"passed": 0, "total": 0},
        "mcq": {"passed": 0, "total": 0},
        "score": {"passed": 0, "total": 0}
    }
    
    # Test with default 'all' variant
    logger.info("=" * 50)
    logger.info("TESTING DEFAULT (ALL) VARIANT")
    logger.info("=" * 50)
    model_all = Model("test:all")
    
    tests = [
        ("Augmented Chunk Generation", test_augmented_chunk_generation),
        ("MCQ Generation", test_mcq_generation),
        ("Verification", test_verification),
        ("Scoring", test_scoring)
    ]
    
    for test_name, test_func in tests:
        results["default"]["total"] += 1
        if test_func(model_all):
            results["default"]["passed"] += 1
    
    # Test with MCQ variant
    logger.info("\n" + "=" * 50)
    logger.info("TESTING MCQ VARIANT")
    logger.info("=" * 50)
    model_mcq = Model("test:mcq")
    
    results["mcq"]["total"] += 1
    if test_mcq_generation(model_mcq):
        results["mcq"]["passed"] += 1
    
    # Test with score variant
    logger.info("\n" + "=" * 50)
    logger.info("TESTING SCORE VARIANT")
    logger.info("=" * 50)
    model_score = Model("test:score")
    
    results["score"]["total"] += 1
    if test_scoring(model_score):
        results["score"]["passed"] += 1
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    for variant, result in results.items():
        logger.info(f"{variant.upper()} variant: {result['passed']}/{result['total']} tests passed")
    
    total_passed = sum(result["passed"] for result in results.values())
    total_tests = sum(result["total"] for result in results.values())
    logger.info(f"OVERALL: {total_passed}/{total_tests} tests passed")
    
    return total_passed == total_tests

if __name__ == "__main__":
    logger.info("Starting test_model_verification script")
    success = run_all_tests()
    
    if success:
        logger.info("All tests passed successfully! The TestModel implementation works correctly.")
        sys.exit(0)
    else:
        logger.error("Some tests failed. Please check the logs for details.")
        sys.exit(1)

