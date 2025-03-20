#!/usr/bin/env python
# make the list of models (defined in config.yml, extracted by src/config.py) 
# availablle to shell scripts

import config

# Print all defined models
if config.defaultModel:
    print(config.defaultModel)
if config.defaultModelB:
    print(config.defaultModelB)
if config.model_c and config.model_c.get("name"):
    print(config.model_c["name"])
if config.model_d and config.model_d.get("name"):
    print(config.model_d["name"])


