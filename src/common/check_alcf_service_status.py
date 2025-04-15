#!/usr/bin/env python

import requests
import sys

from inference_auth_token import get_access_token
alcf_access_token = get_access_token()
from alcf_inference_utilities import get_names_of_alcf_chat_models, get_alcf_inference_service_model_queues
alcf_chat_models = get_names_of_alcf_chat_models(alcf_access_token)


# Define the URL and headers
url = "https://data-portal-dev.cels.anl.gov/resource_server/sophia/jobs"
headers = {
    "Authorization": f"Bearer {alcf_access_token}"
}

# Make the GET request
response = requests.get(url, headers=headers)

# Check the response
if response.status_code == 200:
    status = response.json()
else:
    print("Error:", response.status_code, response.text)
    exit(1)

if len(sys.argv) > 1:
    print(f'Full status: {status}\n')

def get_models(status, status1, status2):
    models = status[status1]
    model_list = []
    for model in models:
        if model['Model Status'] == status2:
            model_list += model['Models Served'].split(',')
    return model_list

models_running = get_models(status, 'running', 'running')
print(f'Running : {models_running}')

models_starting = get_models(status, 'running', 'starting')
print(f'Starting: {models_starting}')

models_queued = get_models(status, 'queued', 'starting')
print(f'Queued  : {models_queued}')

other_models = [model for model in alcf_chat_models if model not in models_queued and model not in models_starting and model not in models_running]
print(f'Inactive: {other_models}')
