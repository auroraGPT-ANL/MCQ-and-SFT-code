#!/usr/bin/env python

import sys
import json
import os
import subprocess
import re
import time
import socket
import requests
import openai
from openai import OpenAI
import logging
from common.config import timeout, logger, initiate_shutdown

from common.exceptions import APITimeoutError

OPENAI_EP  = 'https://api.openai.com/v1'
# Use the Argo endpoint at the /chat level.
ARGO_EP    = 'https://apps.inside.anl.gov/argoapi/api/v1/resource/chat'

class Model:
    def __init__(self, model_name: str):
        """
        Initialize a generic Model object that can handle multiple backends.

        You used to do:
            self.base_model  = config.defaultBaseModel
            self.tokenizer   = config.defaultTokenizer
            self.temperature = config.defaultTemperature
        but now we've removed references to config.default*.

        Instead, we choose a fallback default for each. If you want different
        defaults for different model_name cases, you can add if/else logic here.
        """
        self.model_name   = model_name
        self.base_model   = None        # Will be set if we load HF or something else
        self.tokenizer    = None
        self.temperature  = 0.7         # A fallback default but overridden by config.yml

        self.headers      = {'Content-Type': 'application/json'}
        self.endpoint     = None
        self.model_type   = None
        self.key          = None
        self.client_socket = None
        self.status       = "UNKNOWN"
        self.argo_user    = None        # filled in as needed for Argo model types

        if model_name.startswith('local:'):
            self.model_name = model_name.split('local:')[1]
            logger.info(f"Local model: {model_name}.")
            self.model_type = 'vLLM'
            self.endpoint   = 'http://localhost:8000/v1/chat/completions'

        elif model_name.startswith('pb:'):
            """
            Submit the model job to PBS and store the job ID, then connect
            over sockets. This is specialized code for HPC job submission.
            """
            self.model_name   = model_name.split('pb:')[1]
            logger.info(f"PB model: {model_name}.")
            self.model_type   = 'HuggingfacePBS'
            self.model_script = "run_model.pbs"
            self.job_id       = None
            self.status       = "PENDING"

            logger.info(f'Huggingface model {self.model_name} to be run on HPC system: Starting model server.')
            result = subprocess.run(["qsub", self.model_script], capture_output=True, text=True, check=True)
            self.job_id = result.stdout.strip().split(".")[0]  # Extract job ID
            logger.info(f"Job submitted with ID: {self.job_id}")

            # Wait until job starts running
            self.wait_for_job_to_start()
            self.connect_to_model_server()
            # Send model name to server
            self.client_socket.sendall(self.model_name.encode())
            response = self.client_socket.recv(1024).decode()
            if response != 'ok':
                logger.warning(f"Unexpected response: {response}")
                initiate_shutdown("Initiating shutdown.")
            logger.info("Model server initialized")

        elif model_name.startswith('hf:'):
            """
            'Huggingface' local model approach, reading from huggingface_hub.
            """
            self.model_type = 'Huggingface'
            logger.info(f"HF model: {model_name}.")
            from huggingface_hub import login
            from transformers import AutoModelForCausalLM, AutoTokenizer

            cache_dir = os.getenv("HF_HOME")

            self.model_name = model_name.split('hf:')[1]
            logger.info(f"HF model running locally: {model_name}")
            self.endpoint = 'http://huggingface.co'

            with open("hf_access_token.txt", "r") as file:
                self.key = file.read().strip()
            login(self.key)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                cache_dir=cache_dir
            )
            # Ensure pad/eos tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
            self.tokenizer.padding_side = "right"

        elif model_name.startswith('alcf'):
            """
            Use the ALCF Inference Service endpoint
            """
            self.model_name = model_name.split('alcf:')[1]
            logger.info(f"ALCF (Sophia) Inference Service Model: {self.model_name}")

            from inference_auth_token import get_access_token
            token = get_access_token()
            from alcf_inference_utilities import get_names_of_alcf_chat_models
            alcf_chat_models = get_names_of_alcf_chat_models(token)
            if self.model_name not in alcf_chat_models:
                logger.warning(f"Bad ALCF model: {self.model_name}")
                initiate_shutdown("Initiating shutdown.")
            self.model_type = 'ALCF'
            self.endpoint   = 'https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1'
            self.key        = token

        elif model_name.startswith('cafe'):
            """
            Use Rick's Cafe Inference Service endpoint
            """
            self.model_name = model_name.split('cafe:')[1]
            logger.info(f"Rick's Cafe model: {self.model_name}")

            # For now, simply set the token to a placeholder
            token = "CELS"
            self.model_type = 'CAFE'
            self.endpoint   = 'https://195.88.24.64/v1'
            self.key        = token

        elif model_name.startswith('openai'):
            """
            Use OpenAI's API (e.g. GPT-3.5/4)
            """
            self.model_name = model_name.split('openai:')[1]
            logger.info(f"OpenAI model (run at OpenAI): {self.model_name}")
            self.model_type = 'OpenAI'
            with open('openai_access_token.txt', 'r') as file:
                self.key = file.read().strip()
            self.endpoint = OPENAI_EP

        elif model_name.startswith('argo:'):
            """
            Use Argonne's Argo API service (OpenAI-compatible API)
            """
            from config import argo_user

            self.model_name = model_name.split('argo:')[1]
            logger.info(f"Argo API model: {self.model_name}")
            self.model_type = 'Argo'

            # Get Argonne username from config (which loads from secrets.yml)
            self.argo_user = argo_user
            if not self.argo_user:
                logger.warning("Argo username not found in config/secrets")
                initiate_shutdown("Argo API requires authentication.")

            #logger.info(f"Using Argo user: {self.argo_user}")
            self.endpoint = ARGO_EP
            #logger.info(f"Argo endpoint: {ARGO_EP}")
            # Add dummy API key for OpenAI client compatibility
            self.key = "sk-dummy-key-for-argo-models"

        elif model_name.startswith('test:'):
            """
            Test model type - returns predefined responses for offline testing
            Optional format: test:mcq, test:answer, test:score
            Default is test:all (responds to all types)
            """
            self.model_name = model_name.split('test:')[1] if ':' in model_name else "all"
            logger.info(f"Test model (stub): {self.model_name}")
            self.model_type = 'Test'
            self.endpoint = None  # No endpoint needed
            self.temperature = 0.0  # Deterministic responses
            self.key = None

            # Import the TestModel class here to avoid circular imports
            from test_model import TestModel
            self.test_model = TestModel(self.model_name)

        else:
            logger.warning(f"Bad model: {model_name}")
            initiate_shutdown("Initiating shutdown.")

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
            from transformers import GenerationConfig
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
                response_json = resp.json()
                message = response_json['choices'][0]['message']['content']
                return message
            except Exception as e:
                logger.warning(f"Exception: {str(e)[:80]}...")
                initiate_shutdown("Initiating shutdown.")

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
                from requests.exceptions import Timeout
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
                if "401" in str(e) or "Unauthorized" in str(e):
                    logger.error(f"{self.model_type} authentication failed: {str(e)[:80]}...")
                    initiate_shutdown("Model API Authentication failed. Exiting.")
                logger.warning(f"{self.model_type} request error: {str(e)[:80]}...")
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
                resp = requests.post(url, headers=self.headers, data=json.dumps(params))
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
            except Exception as e:
                logger.warning(f"Argo direct POST request error: {str(e)[:80]}")
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
                logger.info(f"Raw response: {resp}")
                response_json = resp.json()
                logger.info(f"Parsed JSON: {response_json}")
                message = response_json['choices'][0]['message']['content']
                logger.info(f"Response message: {message}")
                return message
            except Exception as e:
                logger.warning(f"Exception: {str(e)[:80]}...")
                initiate_shutdown("Initiating shutdown.")
        elif self.model_type == 'Test':
            logger.info(f"Running test model with prompt: {user_prompt[:50]}...")
            return self.test_model.generate_response(user_prompt, system_prompt)
        else:
            logger.warning(f"Unknown model type: {self.model_type}")
            initiate_shutdown("Initiating shutdown.")

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

