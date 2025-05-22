#!/usr/bin/env python

import json
import logging
from common.config import logger

logger = logging.getLogger(__name__)

class TestModel:
    """
    TestModel provides predefined responses for offline testing.
    It simulates the output of AI models without requiring network connectivity or API keys.

    This class encapsulates all test response logic that was previously in model_access.py,
    allowing for better code organization and easier maintenance of test responses.
    """

    def __init__(self, model_variant="all"):
        """
        Initialize a TestModel instance.

        Args:
            model_variant (str): Optional variant to specialize the model responses.
                                 Values: "mcq", "answer", "score", "nugget", "all" (default)
        """
        self.model_variant = model_variant
        logger.info(f"Test model initialized: variant={model_variant}")

    def generate_response(self, user_prompt, system_prompt=None):
        """
        Generate a predefined response based on the prompt content or model variant.

        Args:
            user_prompt (str): The user prompt/query
            system_prompt (str, optional): System instructions (ignored in test model)

        Returns:
            str: A predefined response in the appropriate format
        """
        # Log detailed information about the request
        logger.info(f"Test model analyzing prompt: first 50 chars = '{user_prompt[:50]}'")

        # 1. Check for augmented chunk generation (summarization)
        if self._is_augmented_chunk_request(user_prompt):
            return self._get_augmented_chunk_response()

        # 2. Check for MCQ generation
        elif self._is_mcq_generation_request(user_prompt):
            return self._get_mcq_generation_response()

        # 3. Check for verification (when we see our own JSON with "question" and "options")
        elif self._is_verification_request(user_prompt):
            return self._get_verification_response()

        # 4. Check for explicit verification request
        elif self._is_explicit_verification_request(user_prompt):
            return self._get_explicit_verification_response()

        # 5. Check for scoring request
        elif self._is_scoring_request(user_prompt):
            return self._get_scoring_response()

        # 6. Check for metadata extraction request (nugget workflow)
        elif self._is_metadata_extraction_request(user_prompt):
            return self._get_metadata_response()

        # 7. Check for fact extraction request (nugget workflow)
        elif self._is_fact_extraction_request(user_prompt):
            return self._get_fact_extraction_response()

        # 8. Check for fact comparison request (nugget workflow)
        elif self._is_fact_comparison_request(user_prompt):
            return self._get_fact_comparison_response()

        # If no conditions match, return default response based on model variant
        return self._get_default_response()

    def _is_augmented_chunk_request(self, prompt):
        """Check if prompt is requesting augmented chunk generation"""
        return ('given the following chunk of text' in prompt.lower() or
                'summarize' in prompt.lower())

    def _is_mcq_generation_request(self, prompt):
        """Check if prompt is requesting MCQ generation"""
        return 'below is some content called augmented_chunk' in prompt.lower()

    def _is_verification_request(self, prompt):
        """Check if prompt contains verification request with JSON"""
        return ('augmented_chunk:' in prompt.lower() and
                '"question"' in prompt and
                '"options"' in prompt)

    def _is_explicit_verification_request(self, prompt):
        """Check if prompt contains explicit verification request"""
        return 'verify' in prompt.lower() and 'score' in prompt.lower()

    def _is_scoring_request(self, prompt):
        """Check if prompt or model variant is for scoring"""
        return ('score' in self.model_variant.lower() or
                'how consistent is the user answer' in prompt.lower())

    def _is_metadata_extraction_request(self, prompt):
        """Check if prompt is requesting paper metadata extraction"""
        return "extract paper metadata" in prompt.lower() or "extract:" in prompt.lower()

    def _is_fact_extraction_request(self, prompt):
        """Check if prompt is requesting fact extraction"""
        patterns = [
            "extract atomic factual statements",
            "factual statements",
            "extract key facts",
            "extract facts"
        ]
        return any(pattern in prompt.lower() for pattern in patterns)

    def _is_fact_comparison_request(self, prompt):
        """Check if prompt is requesting fact comparison"""
        return "compare these two factual statements" in prompt.lower() or "similarity score" in prompt.lower()

    def _get_augmented_chunk_response(self):
        """Return a response for augmented chunk generation"""
        logger.info("MATCHED CONDITION: Augmented chunk generation (step 1)")
        augmented_chunk = {
            "augmented_text": "The test model in this codebase serves as a substitute for real language models during development and testing. It returns pre-defined responses that match the expected format for each script (generate_mcqs.py, generate_answers.py, and score_answers.py), allowing developers to test their code workflow without needing to connect to actual model APIs or services. This significantly speeds up testing and allows for offline development."
        }
        return json.dumps(augmented_chunk)

    def _get_mcq_generation_response(self):
        """Return a response for MCQ generation"""
        logger.info("MATCHED CONDITION: MCQ generation (step 2)")
        mcq_question = {
            "question": "What is the primary purpose of the test model in this codebase?",
            "options": "A. To accelerate model training\nB. To provide offline testing functionality without network calls\nC. To improve code documentation\nD. To connect to remote APIs more efficiently",
            "answer": "B"
        }
        return json.dumps(mcq_question)

    def _get_verification_response(self):
        """Return a verification response"""
        logger.info("MATCHED CONDITION: Verification response (step 3)")
        verification_json = {
            "question": "What is the primary purpose of the test model in this codebase?",
            "answer": "B",
            "score": 9
        }
        return json.dumps(verification_json)

    def _get_explicit_verification_response(self):
        """Return an explicit verification response"""
        logger.info("MATCHED CONDITION: Explicit verification request")
        mcq_json = {
            "question": "What is the primary purpose of the test model in this codebase?",
            "answer": "B",
            "score": 9
        }
        return json.dumps(mcq_json)

    def _get_scoring_response(self):
        """Return a scoring response"""
        logger.info("MATCHED CONDITION: Score request")
        score = 8  # Default good score
        return str(score)

    def _get_metadata_response(self):
        """Return a metadata extraction response"""
        logger.info("MATCHED CONDITION: Metadata extraction request")
        return json.dumps({
            "identifiers": ["10.1234/example.2025.123"],
            "title": "Test Paper Title",
            "first_author": "Test Author"
        })

    def _get_fact_extraction_response(self):
        """Return a fact extraction response with proper claim/span/confidence structure"""
        logger.info("MATCHED CONDITION: Fact extraction request")
        return json.dumps([
            {
                "claim": "Language models have revolutionized NLP",
                "span": "Recent advances in language models have revolutionized natural language processing",
                "confidence": 0.95
            },
            {
                "claim": "GPT models can perform text completion and code generation",
                "span": "GPT models have shown remarkable capabilities in tasks ranging from text completion to code generation",
                "confidence": 0.90
            },
            {
                "claim": "Transformer architectures are used in these models",
                "span": "These models use transformer architectures and are trained on massive datasets",
                "confidence": 0.85
            }
        ])

    def _get_fact_comparison_response(self):
        """Return a fact comparison response"""
        logger.info("MATCHED CONDITION: Fact comparison request")
        return json.dumps({
            "similarity_score": 0.75,
            "reasoning": "The facts share common elements but differ in specific details."
        })

    def _get_default_response(self):
        """Return default response based on model variant"""
        if self.model_variant == "mcq":
            return self._get_mcq_generation_response()
        elif self.model_variant == "answer" or self.model_variant == "verify":
            return self._get_verification_response()
        elif self.model_variant == "score":
            return self._get_scoring_response()
        elif self.model_variant == "nugget":
            return self._get_fact_extraction_response()

        # Default to MCQ format as fallback
        logger.info("Test model using MCQ format as default response")
        return json.dumps({
            "question": "What is the primary purpose of the test model in this codebase?",
            "options": "A. To accelerate model training\nB. To provide offline testing functionality without network calls\nC. To improve code documentation\nD. To connect to remote APIs more efficiently",
            "answer": "B"
        })


