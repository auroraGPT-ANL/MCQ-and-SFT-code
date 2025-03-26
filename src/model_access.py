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
import config

from exceptions import APITimeoutError

logger = logging.getLogger(__name__)

OPENAI_EP  = 'https://api.openai.com/v1'

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
        self.temperature  = 0.7         # A fallback default

        self.headers      = {'Content-Type': 'application/json'}
        self.endpoint     = None
        self.model_type   = None
        self.key          = None
        self.client_socket = None
        self.status       = "UNKNOWN"

        # Identify model type by prefix (local:, hf:, openai:, etc.)
        if model_name.startswith('local:'):
            self.model_name = model_name.split('local:')[1]
            config.logger.info(f"Local model: {model_name}.")
            self.model_type = 'vLLM'
            self.endpoint   = 'http://localhost:8000/v1/chat/completions'

        elif model_name.startswith('pb:'):
            """
            Submit the model job to PBS and store the job ID, then connect
            over sockets. This is specialized code for HPC job submission.
            """
            self.model_name   = model_name.split('pb:')[1]
            self.model_type   = 'HuggingfacePBS'
            self.model_script = "run_model.pbs"
            self.job_id       = None
            self.status       = "PENDING"

            config.logger.info(f'Huggingface model {self.model_name} to be run on HPC system: Starting model server.')
            result = subprocess.run(["qsub", self.model_script], capture_output=True, text=True, check=True)
            self.job_id = result.stdout.strip().split(".")[0]  # Extract job ID
            config.logger.info(f"Job submitted with ID: {self.job_id}")

            # Wait until job starts running
            self.wait_for_job_to_start()
            self.connect_to_model_server()
            # Send model name to server
            self.client_socket.sendall(self.model_name.encode())
            response = self.client_socket.recv(1024).decode()
            if response != 'ok':
                config.logger.warning(f"Unexpected response: {response}")
                config.initiate_shutdown("Initiating shutdown.")
                #sys.exit(1)
            config.logger.info("Model server initialized")

        elif model_name.startswith('hf:'):
            """
            'Huggingface' local model approach, reading from huggingface_hub.
            """
            self.model_type = 'Huggingface'
            from huggingface_hub import login
            from transformers import AutoModelForCausalLM, AutoTokenizer

            cache_dir = os.getenv("HF_HOME")

            self.model_name = model_name.split('hf:')[1]
            config.logger.info(f"HF model running locally: {model_name}")
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
            config.logger.info(f"ALCF Inference Service Model: {self.model_name}")

            from inference_auth_token import get_access_token
            token = get_access_token()
            from alcf_inference_utilities import get_names_of_alcf_chat_models
            alcf_chat_models = get_names_of_alcf_chat_models(token)
            if self.model_name not in alcf_chat_models:
                config.logger.warning(f"Bad ALCF model: {self.model_name}")
                config.initiate_shutdown("Initiating shutdown.")
                #sys.exit(1)
            self.model_type = 'ALCF'
            self.endpoint   = 'https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1'
            self.key        = token

        # models Rick and Tom are running...
        elif model_name.startswith('cafe'):
            """
            Use Rick's Cafe Inference Service endpoint
            """
            self.model_name = model_name.split('cafe:')[1]
            config.logger.info(f"Rick's Cafe Inference Service Model: {self.model_name}")

            # For now, simply set the token to a placeholder
            token = "EMPTY"
            self.model_type = 'CAFE'
            self.endpoint   = 'https://66.55.67.65/v1'
            self.key        = token


        elif model_name.startswith('openai'):
            """
            Use OpenAI's API (e.g. GPT-3.5/4)
            """
            self.model_name = model_name.split('openai:')[1]
            config.logger.info(f"OpenAI model to be run at OpenAI: {self.model_name}")
            self.model_type = 'OpenAI'
            with open('openai_access_token.txt', 'r') as file:
                self.key = file.read().strip()
            self.endpoint = OPENAI_EP

        else:
            config.logger.warning(f"Bad model: {model_name}")
            config.initiate_shutdown("Initiating shutdown.")
            #sys.exit(1)

    def wait_for_job_to_start(self):
        """Monitor job status and get assigned compute node"""
        while True:
            qstat_output = subprocess.run(["qstat", "-f", self.job_id],
                                          capture_output=True, text=True).stdout
            match = re.search(r"exec_host = (\S+)", qstat_output)
            if match:
                self.compute_node = match.group(1).split("/")[0]
                config.logger.info(f"Job {self.job_id} is running on {self.compute_node}")
                self.status = "RUNNING"
                break
            config.logger.info(f"Waiting for job {self.job_id} to start...")
            time.sleep(5)  # Check every 5 seconds

    def connect_to_model_server(self):
        """Establish a persistent TCP connection to the model server"""
        if self.status != "RUNNING":
            raise RuntimeError("Model is not running. Ensure the PBS job is active.")

        config.logger.info(f"Connecting to {self.compute_node} on port 50007...")
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for count in range(10):
            try:
                self.client_socket.connect((self.compute_node, 50007))
                break
            except:
                time.sleep(5)
                config.logger.warning(f"Trying connection again {count}")

    def details(self):
        """
        Print basic info about the model to the logs.
        """
        config.logger.info(f"Model name: {self.model_name}")
        config.logger.info(f"    Model type  = {self.model_type}")
        config.logger.info(f"    Endpoint    = {self.endpoint}")
        config.logger.info(f"    Temperature = {self.temperature}")
        config.logger.info(f"    Base model  = {self.base_model}")
        config.logger.info(f"    Tokenizer   = {self.tokenizer}")

    def close(self):
        """Close the cached connection when done."""
        if self.client_socket:
            config.logger.info("Closing connection to model server.")
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
            # Local HF model
            from transformers import GenerationConfig
            config.logger.info(f"Generating with HF model: {self.model_name}")
            return run_hf_model(user_prompt, self.base_model, self.tokenizer)

        elif self.model_type == 'HuggingfacePBS':
            # HPC job
            if self.status != "RUNNING":
                raise RuntimeError("Model is not running. Ensure the PBS job is active.")
            if self.client_socket is None:
                raise RuntimeError("Socket is not connected")
            config.logger.info(f"Sending input to HPC model: {user_prompt}")
            self.client_socket.sendall(user_prompt.encode())
            response = self.client_socket.recv(1024).decode()
            config.logger.info(f"HPC model response: {response}")
            return response

        elif self.model_type == 'vLLM':
            # local: vLLM-based endpoint
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                "temperature": temperature
            }
            try:
                config.logger.info(f'Running {self.endpoint}\n'
                                   f'  Headers = {self.headers}\n'
                                   f'  Data = {json.dumps(data)}')
                resp = requests.post(self.endpoint, headers=self.headers, data=json.dumps(data))
                config.logger.info(f"Raw response: {resp}")
                response_json = resp.json()
                config.logger.info(f"Parsed JSON: {response_json}")
                message = response_json['choices'][0]['message']['content']
                config.logger.info(f"Response message: {message}")
                return message
            except Exception as e:
                config.logger.warning(f'Exception: {e}')
                config.initiate_shutdown("Initiating shutdown.")
                #sys.exit(1)

        elif self.model_type in ['OpenAI', 'ALCF']:
            # Chat completions via openai or ALCF
            client = OpenAI(
                api_key=self.key,
                base_url=self.endpoint
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ]
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    timeout=60
                )
                generated_text = response.choices[0].message.content.strip()
                return generated_text
            except APITimeoutError as e:
                config.logger.info(f"OpenAI/ALCF request timed out: {e}") #not fatal so don't clutter
                return ""
            except Exception as e:
                if "401" in str(e) or "Unauthorized" in str(e):
                    config.initiate_shutdown("Model API Authentication failed. Exiting.")
                    #sys.exit("Model API Authentication failed. Exiting.")
                config.logger.info(f"OpenAI/ALCF request error: {e}")  # alert the user elsewhere if too many errs
                return ""

        elif self.model_type == 'CAFE':
                # Handle CAFE models similarly to vLLM
                data = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt}
                    ],
                    "temperature": temperature
                }
                try:
                    config.logger.info(
                        f'Running {self.endpoint}\n'
                        f'  Headers = {self.headers}\n'
                        f'  Data = {json.dumps(data)}'
                    )
                    resp = requests.post(self.endpoint, headers=self.headers, data=json.dumps(data))
                    config.logger.info(f"Raw response: {resp}")
                    response_json = resp.json()
                    config.logger.info(f"Parsed JSON: {response_json}")
                    message = response_json['choices'][0]['message']['content']
                    config.logger.info(f"Response message: {message}")
                    return message
                except Exception as e:
                    config.logger.warning(f'Exception: {e}')
                    config.initiate_shutdown("Initiating shutdown.")
                    #sys.exit(1)

        else:
            config.logger.warning(f"Unknown model type: {self.model_type}")
            config.initiate_shutdown("Initiating shutdown.")
            #sys.exit(1)

def run_hf_model(input_text, base_model, tokenizer):
    """
    Generate text from a local HF model.
    Example usage for demonstration purposes only.
    """
    if base_model is None or tokenizer is None:
        return "HF model or tokenizer not loaded."

    # Prepare input for generation
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

