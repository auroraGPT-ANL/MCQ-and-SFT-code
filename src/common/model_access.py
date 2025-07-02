#!/usr/bin/env python3

"""common.model_access
   A flexible wrapper that can talk to multiple back‑ends (OpenAI, Argo, ALCF,
   local vLLM, Hugging‑Face, etc.). Uses settings.endpoints for configuration.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from typing import Any, Optional
import threading

import requests
from huggingface_hub import login
from openai import OpenAI
from requests.exceptions import Timeout
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig)

# Import settings system
from settings import load_settings, Settings
from common.exceptions import APITimeoutError
from common.inference_auth_token import get_access_token
from common.alcf_inference_utilities import get_names_of_alcf_chat_models
from test.test_model import TestModel  # local stub for offline testing

# ---------------------------------------------------------------------------
# Settings initialization
# ---------------------------------------------------------------------------
from pydantic import ValidationError
try:
    settings = load_settings()
except ValidationError as e:
    first = e.errors()[0]
    print(f"❌  {first['msg']}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Local aliases for readability
# ---------------------------------------------------------------------------
logger = getattr(settings, 'logger', None)
if logger is None:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

timeout = settings.timeout
initiate_shutdown = None  # Will be populated from config if needed

# Thread safety for error reporting
error_lock = threading.Lock()


class Model:
    """Unified interface for multiple chat/completions back‑ends."""

    # Class-level error tracking - keyed by provider name from endpoints
    _api_error_counts = {}
    _error_count_lock = threading.Lock()

    # Core Methods
    # -----------------------------------------------------------------------
    def __init__(self, model_name: str, parallel_workers: int = None):
        """Initialize model interface using endpoint configuration."""
        self.model_name = model_name
        self.parallel_workers = parallel_workers
        self.base_model: Any | None = None
        self.tokenizer: Any | None = None
        self.temperature = settings.default_temperature
        self.headers = {"Content-Type": "application/json"}
        self.endpoint: str | None = None
        self.model_type: str | None = None
        self.key: str | None = None
        self.client_socket: socket.socket | None = None
        self.status = "UNKNOWN"
        
        # Special case for test models
        if model_name.startswith("test:"):
            self._init_test_model(model_name)
            return
            
        # Handle provider:model format
        if ":" in model_name:
            provider, model_id = model_name.split(":", 1)
            # Find endpoint by provider
            for name, ep in settings.endpoints.items():
                if ep.provider.value.lower() == provider.lower():
                    self.endpoint_config = ep
                    self.model_name = model_id
                    break
            else:
                logger.error(f"No endpoint configuration found for provider: {provider}")
                if initiate_shutdown:
                    initiate_shutdown(f"Unknown provider: {provider}")
                raise ValueError(f"No endpoint configuration found for provider: {provider}")
        else:
            # Find endpoint by shortname
            shortname = model_name
            for name, ep in settings.endpoints.items():
                if ep.shortname.lower() == shortname.lower():
                    self.endpoint_config = ep
                    self.model_name = ep.model  # Use model from endpoint config
                    break
            else:
                logger.error(f"No endpoint configuration found for model: {model_name}")
                if initiate_shutdown:
                    initiate_shutdown(f"Unknown model: {model_name}")
                raise ValueError(f"No endpoint configuration found for model: {model_name}")
                
        # Set up model based on endpoint configuration
        self.model_type = self.endpoint_config.provider.value.upper()
        self.endpoint = self.endpoint_config.base_url
        
        # Get credential
        cred_key = self.endpoint_config.cred_key
        if cred_key not in settings.secrets:
            logger.error(f"Missing credential: {cred_key}")
            if initiate_shutdown:
                initiate_shutdown(f"Missing credential: {cred_key}")
            raise ValueError(f"Missing credential: {cred_key}")
            
        self.key = settings.secrets[cred_key].get_secret_value()
        
        # Special handling for provider types
        provider = self.endpoint_config.provider.value
        
        if provider == "hf":
            self._setup_huggingface()
        elif provider == "alcf":
            self._setup_alcf()
            
        logger.info(f"Initialized {self.model_type} model: {self.model_name}")

    def _setup_huggingface(self):
        """Special setup for HuggingFace models."""
        login(self.key)
        cache_dir = os.getenv("HF_HOME")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.endpoint_config.model, 
            cache_dir=cache_dir
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.endpoint_config.model,
            device_map="auto",
            cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "right"

    def _setup_alcf(self):
        """Special setup for ALCF models."""
        token = get_access_token()
        if self.model_name not in get_names_of_alcf_chat_models(token):
            logger.error(f"ALCF model {self.model_name} not in registry")
            if initiate_shutdown:
                initiate_shutdown(f"ALCF model {self.model_name} not in registry")
            raise ValueError(f"ALCF model {self.model_name} not in registry")

    # "test:<model>" - Local stub model for offline testing
    def _init_test_model(self, model_name: str):
        self.model_name = model_name.split("test:")[1] if ":" in model_name else "all"
        self.model_type = "Test"
        self.temperature = 0.0
        self.test_model = TestModel(self.model_name)
        logger.info(f"Loaded TestModel stub ({self.model_name})")

    # No longer needed PBS helper methods have been removed
    def details(self):
        """Print basic info about the model to the logs."""
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"    Model type  = {self.model_type}")
        logger.info(f"    Endpoint    = {self.endpoint}")
        logger.info(f"    Temperature = {self.temperature}")
        if self.base_model:
            logger.info(f"    Base model  = {self.base_model}")
        if self.tokenizer:
            logger.info(f"    Tokenizer   = {self.tokenizer}")

    def run(
        self,
        user_prompt='Tell me something interesting',
        system_prompt='You are a helpful assistant',
        temperature=None
    ) -> str:
        """Generate response using the configured model."""
        if temperature is None:
            temperature = self.temperature

        # Handle test models
        if self.model_type == 'TEST':
            return self.test_model.generate_response(user_prompt, system_prompt)

        # Handle HuggingFace models
        if self.model_type == 'HF':
            return run_hf_model(user_prompt, self.base_model, self.tokenizer)

        # All other models use API requests
        try:
            # Prepare base request data
            data = self._format_chat_request(user_prompt, system_prompt, temperature)

            # Add provider-specific parameters
            if self.model_type in ['ARGO', 'ARGO_DEV']:
                argo_username = settings.secrets.get("argo_username")
                if argo_username:
                    data.update({
                        "user": argo_username.get_secret_value(),
                        "stop": [],
                        "top_p": 0.9
                    })

            # Handle OpenAI-style APIs
            if self.model_type in ['OPENAI', 'ALCF']:
                return self._handle_openai_request(data)

            # Handle other API-based models
            return self._handle_api_model_request(data)

        except Exception as e:
            return self._handle_api_error(e)

    # Error Handling Methods
    # -----------------------------------------------------------------------
    def _get_error_threshold(self) -> int:
        """Get error threshold based on number of worker threads"""
        # n+1 where n is number of threads, default to n=4 if not set
        workers = self.parallel_workers or 4
        return workers + 1

    def _reset_api_error_count(self):
        """Reset error count after successful API call."""
        with self._error_count_lock:
            if self.model_type in self._api_error_counts:
                self._api_error_counts[self.model_type] = 0

    def _increment_api_error_count(self) -> bool:
        """
        Increment the error count for this model type.
        Returns True if we've hit the threshold (n+1 errors before first success).
        """
        with self._error_count_lock:
            if self.model_type not in self._api_error_counts:
                return True  # Non-tracked model types exit immediately

            self._api_error_counts[self.model_type] += 1
            threshold = self._get_error_threshold()
            current = self._api_error_counts[self.model_type]
            logger.debug(f"Error count for {self.model_type}: {current}/{threshold}")
            return current >= threshold

    def _handle_api_error(self, error: Exception) -> str:
        """Thread-safe handling of API errors."""
        error_str = str(error)
        with error_lock:
            # Auth failures always exit immediately
            if "401" in error_str or "Unauthorized" in error_str:
                logger.error(f"{self.model_type} authentication failed: {error_str[:80]}...")
                raise ValueError(f"Authentication failed for {self.model_type}")

            # For API errors (4xx), track and possibly exit
            if "404" in error_str or any(str(code) in error_str for code in range(400, 500)):
                if self._increment_api_error_count():
                    logger.error(f"{self.model_type} API error threshold reached: {error_str[:80]}...")
                    raise ValueError(f"{self.model_type} API errors exceeded threshold")
            else:
                # Non-API errors get warned
                logger.warning(f"{self.model_type} request error: {error_str[:80]}...")

        return ""

    # API Request Helpers
    # -----------------------------------------------------------------------
    def _create_http_session(self) -> requests.Session:
        """Create an HTTP session with configured retry and timeout settings."""
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=settings.http_client.max_retries,
            pool_connections=settings.http_client.pool_connections,
            pool_maxsize=settings.http_client.pool_maxsize
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _make_api_request(self, url: str, data: dict) -> Optional[requests.Response]:
        """Make an API request with proper error handling and timeout settings."""
        session = self._create_http_session()
        if not session:
            return None

        connect_timeout = settings.http_client.connect_timeout
        read_timeout = settings.http_client.read_timeout

        try:
            resp = session.post(
                url,
                headers=self.headers,
                data=json.dumps(data),
                timeout=(connect_timeout, read_timeout)
            )
            if resp.status_code >= 400 and resp.status_code < 500:
                self._handle_api_error(Exception(f"Error code: {resp.status_code} - {resp.text}"))
                return None
            resp.raise_for_status()
            self._reset_api_error_count()  # Success! Reset error count
            return resp
        except requests.exceptions.ConnectionError as e:
            return self._handle_api_error(e)
        except Exception as e:
            return self._handle_api_error(e)

    def _format_chat_request(self, user_prompt: str, system_prompt: str, temperature: float) -> dict:
        """Format a standard chat request payload."""
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature
        }

    def _normalize_url(self, url: str) -> str:
        """Ensure URL ends with a trailing slash."""
        return url if url.endswith('/') else url + '/'

    def _extract_response_content(self, response_json: dict) -> str:
        """Extract the generated text from a response in a standard way."""
        if "choices" in response_json:
            return response_json['choices'][0]['message']['content'].strip()
        elif "response" in response_json:
            return self._parse_argo_response(response_json["response"])
        return str(response_json)

    def _parse_argo_response(self, response_text: str) -> str:
        """Parse Argo-specific response format."""
        if response_text.startswith("```json"):
            response_text = response_text[len("```json"):].strip()
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()
            try:
                parsed = json.loads(response_text)
                return parsed.get("answer", str(parsed)).strip()
            except Exception as parse_error:
                logger.warning(f"Failed to parse JSON from Argo response: {parse_error}")
        return response_text
    def _handle_api_model_request(self, data: dict) -> str:
        """Common handler for API-based models."""
        resp = self._make_api_request(self._normalize_url(self.endpoint), data)
        if resp is None:
            return ""

        try:
            return self._extract_response_content(resp.json())
        except Exception as e:
            return self._handle_api_error(e)

    def _handle_openai_request(self, data: dict) -> str:
        """Common handler for OpenAI-compatible APIs."""
        try:
            client = OpenAI(
                api_key=self.key,
                base_url=self.endpoint,
                timeout=timeout,
                max_retries=1
            )
            try:
                response = client.chat.completions.create(**data)
                self._reset_api_error_count()
                return response.choices[0].message.content.strip()
            except (Timeout, APITimeoutError) as e:
                return self._handle_api_error(e)
            except Exception as e:
                return self._handle_api_error(e)
        except Exception as e:
            return self._handle_api_error(e)

def run_hf_model(input_text, base_model, tokenizer):
    """
    Generate text from a local HF model.
    Example usage for demonstration purposes only.
    """
    if base_model is None or tokenizer is None:
        return "HF model or tokenizer not loaded."

    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    output = base_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=512,
        pad_token_id=tokenizer.pad_token_id
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

