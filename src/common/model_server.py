#!/usr/bin/env python

import socket
import time
import os

HOST = socket.gethostname()
PORT = 50007

cache_dir = os.getenv("HF_HOME")

print(f"Model server running on {HOST}:{PORT} with cache directory {cache_dir}")

# Create a socket (TCP)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)

print("HFMS: Waiting for incoming connections...")

conn, addr = server_socket.accept()
print(f"HFMS: Connected by {addr}")

model_type = 'Huggingface'

from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

model_name = conn.recv(1024).decode()
if not model_name:
    print("HFMS: No modelname received")
    exi(1)

print(f"HFMS: Received model name: {model_name}")

# model_name = 'argonne-private/AuroraGPT-IT-v4-0125'
print('HF model:', model_name)
endpoint = 'http://huggingface.co'

with open("hf_access_token.txt", "r") as file:
    key = file.read().strip()
login(key)

max_seq_length = 2048  # e.g. for models that support RoPE scaling

# Define the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# Load base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    cache_dir=cache_dir
)

# Ensure pad token is set correctly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Ensure model has correct pad token ID
base_model.config.pad_token_id = tokenizer.pad_token_id

# Ensure tokenizer uses correct padding side
tokenizer.padding_side = "right"  # Recommended for LLaMA

response = 'ok'
conn.sendall(response.encode())

def run_hf_model(input_text, base_model, tokenizer):
    # Prepare input for generation
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    # Explicitly move tensors to GPU
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")  # Pass attention mask

    # Explicitly pass attention_mask
    output = base_model.generate(input_ids, attention_mask=attention_mask, max_length=512, pad_token_id=tokenizer.pad_token_id)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


while True:
    try:
        # Receive input data
        question = conn.recv(1024).decode()
        if not question:
            print("HFMS: No question received")
            break  # Client disconnected

        #print(f"HFMS: Received request: {question}")
        response = run_hf_model(question, base_model, tokenizer)
        #print('HFMS: Response =', response)

        # Send response back
        conn.sendall(response.encode())

    except ConnectionResetError:
        print("HFMS: Client disconnected.")
        break

# Cleanup
conn.close()
server_socket.close()
print("Server shutting down.")

