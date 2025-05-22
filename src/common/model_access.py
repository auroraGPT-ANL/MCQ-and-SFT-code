#!/usr/bin/env python

"""common.model_access
   A flexible wrapper that can talk to multiple back‑ends (OpenAI, Argo, ALCF,
   local vLLM, Hugging‑Face, etc.).  Refactored to import the whole *config*
   module so new secrets / parameters can be accessed without editing this file.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from typing import Any
import threading

import requests
from huggingface_hub import login
from openai import OpenAI
from requests.exceptions import Timeout
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig)

# Import new settings system
from common.settings import load_settings
from common import config  # Keep for backward compatibility
from common.inference_auth_token import get_access_token
from common.exceptions import APITimeoutError
from common.alcf_inference_utilities import get_names_of_alcf_chat_models
from test.test_model import TestModel  # local stub for offline testing

# ---------------------------------------------------------------------------
# Settings initialization
# ---------------------------------------------------------------------------
settings = load_settings()

# ---------------------------------------------------------------------------
# Local aliases for readability
# ---------------------------------------------------------------------------
logger = config.logger  # Keep logger from config module
timeout = settings.timeout
initiate_shutdown = config.initiate_shutdown  # Keep shutdown handling from config
# Get argo_user from settings if available, otherwise from config for backward compatibility
argo_user = settings.secrets.get("argo_username", config.argo_user)

# Thread safety for error reporting
error_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class Model:
    """Unified interface for multiple chat/completions back‑ends."""

    # Class-level error tracking
    _api_error_counts = {
        'OpenAI': 0,
        'Argo': 0,
        'CAFE': 0
    }
    _error_count_lock = threading.Lock()  # Lock for thread-safe counter updates

    # Core Methods
    # -----------------------------------------------------------------------
    def __init__(self, model_name: str, parallel_workers: int = None):
        self.model_name = model_name
        self.parallel_workers = parallel_workers
        self.base_model: Any | None = None
        self.tokenizer: Any | None = None
        self.temperature = 0.7

        self.headers = {"Content-Type": "application/json"}
        self.endpoint: str | None = None
        self.model_type: str | None = None
        self.key: str | None = None
        self.client_socket: socket.socket | None = None
        self.status = "UNKNOWN"
        self.argo_user: str | None = None

        # Branch on prefix -----------------------------------------------------
        if model_name.startswith("local:"):
            # ------------------------------------------------------------------
            # Local vLLM server
            # ------------------------------------------------------------------
            self.model_name = model_name.split("local:")[1]
            self.model_type = "vLLM"
            self.endpoint = "http://localhost:8000/v1/chat/completions"
            logger.info(f"Local vLLM model: {self.model_name}")

        elif model_name.startswith("pb:"):
            # ------------------------------------------------------------------
            # PBS‑launched Hugging‑Face model on HPC
            # ------------------------------------------------------------------
            self._init_pbs_model(model_name)

        elif model_name.startswith("hf:"):
            # ------------------------------------------------------------------
            # Local Hugging Face model
            # ------------------------------------------------------------------
            self._init_hf_model(model_name)

        elif model_name.startswith("alcf:"):
            self._init_alcf_model(model_name)

        elif model_name.startswith("cafe:"):
            self._init_cafe_model(model_name)

        elif model_name.startswith("openai:"):
            self._init_openai_model(model_name)

        elif model_name.startswith("argo:") or model_name.startswith("argo-dev:"):
            self._init_argo_model(model_name)

        elif model_name.startswith("test:"):
            self._init_test_model(model_name)

        else:
            logger.error(f"Unrecognised model specifier: {model_name}")
            initiate_shutdown("Model initialisation failed.")

    # -----------------------------------------------------------------------
    #  Initialiser helpers
    # -----------------------------------------------------------------------
    def _init_pbs_model(self, model_name: str):
        self.model_name = model_name.split("pb:")[1]
        self.model_type = "HuggingfacePBS"
        self.model_script = "run_model.pbs"
        self.status = "PENDING"
        logger.info(f"Submitting Hugging‑Face PBS job for {self.model_name}")

        # Submit and capture job‑ID
        result = subprocess.run(["qsub", self.model_script], capture_output=True, text=True, check=True)
        self.job_id = result.stdout.strip().split(".")[0]
        self.wait_for_job_to_start()
        self.connect_to_model_server()
        self.client_socket.sendall(self.model_name.encode())
        if self.client_socket.recv(1024).decode() != "ok":
            initiate_shutdown("Model server init handshake failed.")
        logger.info("Model server initialised.")

    # "hf:<model>" - HF
    def _init_hf_model(self, model_name: str):
        self.model_type = "Huggingface"
        self.model_name = model_name.split("hf:")[1]
        logger.info(f"Loading local HF model: {self.model_name}")

        cache_dir = os.getenv("HF_HOME")
        # Prefer secret token from settings; fall back to config and file for legacy setups
        self.key = settings.secrets.get("huggingface_token") or config.get_secret("huggingface.token")
        if not self.key:
            try:
                with open("hf_access_token.txt", "r", encoding="utf-8") as fh:
                    self.key = fh.read().strip()
            except FileNotFoundError:
                initiate_shutdown("Missing Hugging Face access token.")

        login(self.key)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", cache_dir=cache_dir)
        # Ensure pad/eos tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "right"
        self.endpoint = "http://huggingface.co"

    # "alcf:<model>" - ALCF
    def _init_alcf_model(self, model_name: str):
        token = get_access_token()
        self.model_name = model_name.split("alcf:")[1]
        if self.model_name not in get_names_of_alcf_chat_models(token):
            initiate_shutdown(f"ALCF model {self.model_name} not in registry.")
        self.model_type = "ALCF"
        self.endpoint = "https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1"
        self.key = token
        logger.info(f"ALCF model selected: {self.model_name}")

    # "cafe:<model>" - Rick's secret server
    def _init_cafe_model(self, model_name: str):
        self.model_name = model_name.split("cafe:")[1]
        self.model_type = "CAFE"
        self.endpoint = "https://195.88.24.64/v1"
        self.key = "CELS"  # placeholder
        logger.info(f"Cafe model: {self.model_name}")

    # "openai:<model>" - OpenAI
    def _init_openai_model(self, model_name: str):
        self.model_name = model_name.split("openai:")[1]
        self.model_type = "OpenAI"
        # Try to get endpoint from settings, fall back to config for backward compatibility
        self.endpoint = settings.endpoints.get("openai", {}).get("base_url") if "openai" in settings.endpoints else config.model_type_endpoints['openai']

        # look first in settings.secrets; fall back to config and then to openai_access_token.txt
        self.key = settings.secrets.get("openai_access_token") or config.get_secret("openai.access_token")
        if not self.key:
            try:
                with open("openai_access_token.txt", "r", encoding="utf-8") as fh:
                    self.key = fh.read().strip()
                logger.info(
                    "OpenAI access token found in openai_access_token.txt - please move it to secrets.yml\n"
                    "YAML format:\n"
                    "openai:\n"
                    "      access_token: YOUR_OPENAI_ACCESS_TOKEN"
                )
            except FileNotFoundError:
                initiate_shutdown("Missing OpenAI access token.")
        logger.info(f"OpenAI model: {self.model_name}")


    # "argo:<model>" - Argonne ARGO
    def _init_argo_model(self, model_name: str):
        # Split off prefix and identify if its dev or standard argo
        prefix, model = model_name.split(":")
        self.model_name = model
        self.model_type = "Argo"
        # Select endpoint based on prefix, using settings if available
        endpoint_key = "argo_dev" if prefix == "argo-dev" else "argo"
        if endpoint_key in settings.endpoints:
            self.endpoint = settings.endpoints[endpoint_key].base_url
        else:
            # Fall back to config for backward compatibility
            self.endpoint = config.model_type_endpoints['argo_dev'] if prefix == "argo-dev" else config.model_type_endpoints['argo']

        self.argo_user = settings.secrets.get("argo_username") or argo_user or config.get_secret("argo.username")
        if not self.argo_user:
            initiate_shutdown("Argo username not found in settings or secrets.yml.")
        self.key = "sk‑dummy‑key"  # placeholder for OpenAI‑compatible header
        logger.info(f"Argo model: {self.model_name} (user = {self.argo_user})")

    # "test:<model>" - Local stub model for offline testing
    def _init_test_model(self, model_name: str):
        self.model_name = model_name.split("test:")[1] if ":" in model_name else "all"
        self.model_type = "Test"
        self.temperature = 0.0
        self.test_model = TestModel(self.model_name)
        logger.info(f"Loaded TestModel stub ({self.model_name})")

    # -----------------------------------------------------------------------
    #  PBS helper methods
    # -----------------------------------------------------------------------
    def wait_for_job_to_start(self):
        """Monitor job status and get assigned compute node"""
        while True:
            qstat_output = subprocess.run(["qstat", "-f", self.job_id],
                                          capture_output=True, text=True).stdout
            match = re.search(r"exec_host = (\S+)", qstat_output)
            if match:
                self.compute_node = match.group(1).split("/")[0]
                logger.info(f"Job {self.job_id} is running on {self.compute_node}")
                self.status = "RUNNING"
                break
            logger.info(f"Waiting for job {self.job_id} to start...")
            time.sleep(5)  # Check every 5 seconds

    def connect_to_model_server(self):
        """Establish a persistent TCP connection to the model server"""
        if self.status != "RUNNING":
            raise RuntimeError("Model is not running. Ensure the PBS job is active.")

        logger.info(f"Connecting to {self.compute_node} on port 50007...")
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for count in range(10):
            try:
                self.client_socket.connect((self.compute_node, 50007))
                break
            except Exception as e:
                if count < 9:  # Don't sleep on last attempt
                    time.sleep(5)
                    logger.warning(f"Trying connection again {count}")
                else:
                    return self._handle_api_error(e, " (connection failed)")
    def details(self):
        """
        Print basic info about the model to the logs.
        """
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"    Model type  = {self.model_type}")
        logger.info(f"    Endpoint    = {self.endpoint}")
        logger.info(f"    Temperature = {self.temperature}")
        logger.info(f"    Base model  = {self.base_model}")
        logger.info(f"    Tokenizer   = {self.tokenizer}")

    def close(self):
        """Close the cached connection when done."""
        if self.client_socket:
            logger.info("Closing connection to model server.")
            self.client_socket.close()
            self.client_socket = None

    def run(
        self,
        user_prompt='Tell me something interesting',
        system_prompt='You are a helpful assistant',
        temperature=None
    ) -> str:
        """
        Calls this model to generate an answer to the user_prompt.
        If temperature is not specified, we default to self.temperature.
        """
        if temperature is None:
            temperature = self.temperature  # fallback

        if self.model_type == 'Huggingface':
            #logger.info(f"Generating with HF model: {self.model_name}")
            return run_hf_model(user_prompt, self.base_model, self.tokenizer)

        elif self.model_type == 'HuggingfacePBS':
            if self.status != "RUNNING":
                raise RuntimeError("Model is not running. Ensure the PBS job is active.")
            if self.client_socket is None:
                raise RuntimeError("Socket is not connected")
            #logger.info(f"Sending input to HPC model: {user_prompt}")
            self.client_socket.sendall(user_prompt.encode())
            response = self.client_socket.recv(1024).decode()
            #logger.info(f"HPC model response: {response}")
            return response

        elif self.model_type == 'vLLM':
            return self._handle_api_model_request(user_prompt, system_prompt, temperature)

        elif self.model_type in ['OpenAI', 'ALCF']:
            return self._handle_openai_request(user_prompt, system_prompt, temperature)

        elif self.model_type == 'Argo':
            extra_data = {
                "user": self.argo_user,
                "stop": [],
                "top_p": 0.9
            }
            return self._handle_api_model_request(user_prompt, system_prompt, temperature, extra_data)

        elif self.model_type == 'CAFE':
            return self._handle_api_model_request(user_prompt, system_prompt, temperature)

        elif self.model_type == 'Test':
            return self.test_model.generate_response(user_prompt, system_prompt)

        else:
            logger.error(f"Unknown model type: {self.model_type}.")
            initiate_shutdown("Unknown model type. Exiting.")

    # Error Handling Methods
    # -----------------------------------------------------------------------
    def _get_error_threshold(self) -> int:
        """Get error threshold based on number of worker threads"""
        try:
            # n+1 where n is number of threads, default to n=4 if not set
            workers = (
                self.parallel_workers
                if self.parallel_workers is not None
                else getattr(config, 'parallel_workers', 4)
            )
            return workers + 1
        except AttributeError:
            return 5  # reasonable default if parallel_workers not set

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

    def _handle_api_error(self, error: Exception, context: str = "") -> str:
        """Thread-safe handling of API errors."""
        error_str = str(error)
        with error_lock:
            if not hasattr(config, 'shutdown_event') or not config.shutdown_event.is_set():
                # Auth failures always exit immediately
                if "401" in error_str or "Unauthorized" in error_str:
                    logger.error(f"{self.model_type} authentication failed: {error_str[:80]}...")
                    initiate_shutdown("Model API Authentication failed. Exiting.")
                # For API errors (4xx), only show error and exit if we hit threshold
                elif "404" in error_str or any(str(code) in error_str for code in range(400, 500)):
                    if self._increment_api_error_count():
                        logger.error(f"{self.model_type} API error threshold reached: {error_str[:80]}...")
                        initiate_shutdown(f"{self.model_type} API errors exceeded threshold. Exiting.")
                else:
                    # Non-API errors still get warned
                    logger.info(f"{self.model_type} request error: {error_str[:80]}...")
        return ""

    # API Request Helpers
    # -----------------------------------------------------------------------
    def _create_http_session(self) -> requests.Session:
        """Create an HTTP session with configured retry and timeout settings."""
        if hasattr(config, 'shutdown_event') and config.shutdown_event.is_set():
            return None

        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=settings.http_client.max_retries,
            pool_connections=settings.http_client.pool_connections,
            pool_maxsize=settings.http_client.pool_maxsize
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _make_api_request(self, url: str, data: dict) -> requests.Response:
        """Make an API request with proper error handling and timeout settings."""
        if config.shutdown_event.is_set():
            return None

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
                return self._handle_api_error(Exception(f"Error code: {resp.status_code} - {resp.text}"))
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

    def _handle_api_model_request(self, user_prompt: str, system_prompt: str, temperature: float, extra_data: dict = None) -> str:
        """Common handler for API-based models (vLLM, Argo, CAFE)"""
        data = self._format_chat_request(user_prompt, system_prompt, temperature)
        if extra_data:
            data.update(extra_data)

        resp = self._make_api_request(self._normalize_url(self.endpoint), data)
        if resp is None:
            return ""

        try:
            return self._extract_response_content(resp.json())
        except Exception as e:
            return self._handle_api_error(e)

    def _handle_openai_request(self, user_prompt: str, system_prompt: str, temperature: float) -> str:
        """Common handler for OpenAI-compatible APIs (OpenAI, ALCF)"""
        try:
            params = self._format_chat_request(user_prompt, system_prompt, temperature)
            client = OpenAI(
                api_key=self.key,
                base_url=self.endpoint,
                timeout=timeout,
                max_retries=1
            )
            try:
                response = client.chat.completions.create(**params)
                self._reset_api_error_count()  # Success! Reset error count
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

