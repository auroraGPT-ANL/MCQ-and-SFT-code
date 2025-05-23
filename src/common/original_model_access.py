#!/usr/bin/env python

"""common.model_access
   A flexible wrapper that can talk to multiple back‑ends (OpenAI, Argo, ALCF,
   local vLLM, Hugging‑Face, etc.).  Refactored to import the whole *config*
   module so new secrets / parameters can be accessed without editing this file.

   This version of the code is prior to a major reorganization and introduction of
   helper functions rather than duplicating logic for each model type.  The 
   reorganized version also implements an error tolerance that initiates a
   shutdown after it becomes clear that the interacion with the model isn't working
   (e.g., 404 not found, time-outs), wherease the original version here just
   reports the problem and continues happily.
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

from common import config
from common.inference_auth_token import get_access_token
from common.exceptions import APITimeoutError
from common.alcf_inference_utilities import get_names_of_alcf_chat_models
from test.test_model import TestModel  # local stub for offline testing

# ---------------------------------------------------------------------------
# Local aliases for readability 
# ---------------------------------------------------------------------------
logger = config.logger
timeout = config.timeout
initiate_shutdown = config.initiate_shutdown
argo_user = config.argo_user

# Thread safety for error reporting
error_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Constants 
# ---------------------------------------------------------------------------


class Model:
    """Unified interface for multiple chat/completions back‑ends."""

    def __init__(self, model_name: str):
        self.model_name = model_name  # may be rewritten below
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
    
    # "pb:<model>" - PBS submission
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
        # Prefer secret token; fall back to file for legacy setups
        self.key = config.get_secret("huggingface.token")
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
    def _orig__init_openai_model(self, model_name: str):
        self.model_name = model_name.split("openai:")[1]
        self.model_type = "OpenAI"
        self.endpoint = OPENAI_EP
        try:
            with open("openai_access_token.txt", "r", encoding="utf-8") as fh:
                self.key = fh.read().strip()
        except FileNotFoundError:
            initiate_shutdown(f"Missing OpenAI access token. Provide in {os.getcwd()}/openai_access_token.txt")
        logger.info(f"OpenAI model: {self.model_name}")
        # "openai:<model>" - OpenAI

    def _init_openai_model(self, model_name: str):
        self.model_name = model_name.split("openai:")[1]
        self.model_type = "OpenAI"
        self.endpoint = config.model_type_endpoints['openai']

        # look first in secrets.yml; fall back to openai_access_token.txt and nudge the user to use secrets.yml
        self.key = config.get_secret("openai.access_token")
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
        # Select endpoint based on prefix
        self.endpoint = config.model_type_endpoints['argo_dev'] if prefix == "argo-dev" else config.model_type_endpoints['argo']
        self.argo_user = argo_user or config.get_secret("argo.username")
        if not self.argo_user:
            initiate_shutdown("Argo username not found in secrets.yml.")
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
            except:
                time.sleep(5)
                logger.warning(f"Trying connection again {count}")

    # -----------------------------------------------------------------------
    #  General methods 
    # -----------------------------------------------------------------------
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
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature
            }
            try:
                #logger.info(f'Running {self.endpoint} ')
                resp = requests.post(self.endpoint, headers=self.headers, data=json.dumps(data))
                if resp.status_code >= 400 and resp.status_code < 500:
                    logger.error(f"Model API error {resp.status_code}: {resp.text}")
                    initiate_shutdown(f"Model API error {resp.status_code}. Exiting.")
                response_json = resp.json()
                message = response_json['choices'][0]['message']['content']
                return message
            except Exception as e:
                logger.error(f"Exception: {str(e)[:80]}...")
                initiate_shutdown("Model invocation failed. Exiting.")

        elif self.model_type in ['OpenAI', 'ALCF']:
            #logger.info(f"Initializing {self.model_type} client with model {self.model_name} and endpoint: {self.endpoint}")
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature
                }
                client = OpenAI(
                    api_key=self.key,
                    base_url=self.endpoint,
                    timeout=timeout,
                    max_retries=1
                )
                #logger.info(f"Sending request to {self.model_type} endpoint...")
                #logger.info(f"Request details: model={self.model_name}, temperature={temperature}")
                try:
                    response = client.chat.completions.create(**params)
                    #logger.info("Response received from API")
                    generated_text = response.choices[0].message.content.strip()
                    return generated_text
                except Timeout:
                    logger.warning(f"{self.model_type} request timed out after {timeout} seconds")
                    return ""
                except APITimeoutError as e:
                    logger.warning(f"{self.model_type} request timed out after {timeout} seconds: {str(e)[:80]}...")
                    return ""
            except Exception as e:
                error_str = str(e)
                if "401" in error_str or "Unauthorized" in error_str:
                    logger.error(f"{self.model_type} authentication failed: {error_str[:80]}...")
                    initiate_shutdown("Model API Authentication failed. Exiting.")
                elif "404" in error_str:
                    logger.error(f"{self.model_type} model not found: {error_str[:80]}...")
                    initiate_shutdown(f"{self.model_type} model not found. Exiting.")
                elif any(str(code) in error_str for code in range(400, 500)):
                    logger.error(f"{self.model_type} API error: {error_str[:80]}...")
                    initiate_shutdown(f"{self.model_type} API error. Exiting.")
                logger.warning(f"{self.model_type} request error: {error_str[:80]}...")
                return ""

        elif self.model_type == 'Argo':
            # Direct POST using requests for Argo.
            #logger.info(f"Direct POST to Argo: {self.model_name} at {self.endpoint}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "user": self.argo_user,
                "stop": [],
                "top_p": 0.9
            }
            url = self.endpoint if self.endpoint.endswith('/') else self.endpoint + '/'

            try:
                if config.shutdown_event.is_set():
                    return ""
                    
                # Create a session with custom retry settings from config
                session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    max_retries=config.http_client.get("max_retries", 1),
                    pool_connections=config.http_client.get("pool_connections", 1),
                    pool_maxsize=config.http_client.get("pool_maxsize", 1)
                )
                session.mount('http://', adapter)
                session.mount('https://', adapter)
                
                # Make request with timeouts from config
                connect_timeout = config.http_client.get("connect_timeout", 3.05)
                read_timeout = config.http_client.get("read_timeout", 10)
                resp = session.post(
                    url,
                    headers=self.headers,
                    data=json.dumps(params),
                    timeout=(connect_timeout, read_timeout)
                )
                if resp.status_code >= 400 and resp.status_code < 500:
                    logger.error(f"Model API error {resp.status_code}: {resp.text}")
                    initiate_shutdown(f"Model API error {resp.status_code}. Exiting.")
                resp.raise_for_status()
                response_json = resp.json()
                if "choices" in response_json:
                    generated_text = response_json['choices'][0]['message']['content'].strip()
                    return generated_text
                elif "response" in response_json:
                    generated_text = response_json["response"]
                    # Remove markdown formatting if present.
                    if generated_text.startswith("```json"):
                        generated_text = generated_text[len("```json"):].strip()
                        if generated_text.endswith("```"):
                            generated_text = generated_text[:-3].strip()
                        try:
                            parsed = json.loads(generated_text)
                            if "answer" in parsed:
                                generated_text = parsed["answer"].strip()
                            else:
                                generated_text = str(parsed)
                        except Exception as parse_error:
                            logger.warning(f"Failed to parse JSON from Argo response: {parse_error}")
                    return generated_text
                else:
                    logger.warning(f"Argo response does not contain 'choices' or 'response': {response_json}")
                    return str(response_json)
            except requests.exceptions.ConnectionError as e:
                with error_lock:
                    if not config.shutdown_event.is_set():
                        logger.error(f"Failed to connect to Argo service: {str(e)[:80]}...")
                        initiate_shutdown("Argo service unreachable. Exiting.")
                return ""
            except Exception as e:
                with error_lock:
                    if not config.shutdown_event.is_set():
                        logger.error(f"Argo request error: {str(e)[:80]}...")
                        initiate_shutdown("Argo request failed. Exiting.")
                return ""

        elif self.model_type == 'CAFE':
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature
            }
            try:
                logger.info(
                    f'Running {self.endpoint}\n'
                    f'  Headers = {self.headers}\n'
                    f'  Data = {json.dumps(data)}'
                )
                resp = requests.post(self.endpoint, headers=self.headers, data=json.dumps(data))
                if resp.status_code >= 400 and resp.status_code < 500:
                    logger.error(f"Model API error {resp.status_code}: {resp.text}")
                    initiate_shutdown(f"Model API error {resp.status_code}. Exiting.")
                logger.info(f"Raw response: {resp}")
                response_json = resp.json()
                logger.info(f"Parsed JSON: {response_json}")
                message = response_json['choices'][0]['message']['content']
                logger.info(f"Response message: {message}")
                return message
            except Exception as e:
                logger.error(f"Exception: {str(e)[:80]}...")
                initiate_shutdown("Error invoking cafe type model. Exiting.")
        elif self.model_type == 'Test':
            return self.test_model.generate_response(user_prompt, system_prompt)
        else:
            logger.error(f"Unknown model type: {self.model_type}.")
            initiate_shutdown ("Unknown model type. Exiting.")

    # -----------------------------------------------------------------------
    #  HuggingFace
    # -----------------------------------------------------------------------

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
