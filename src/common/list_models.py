#!/usr/bin/env python
# make the list of models (defined in config.yml, extracted by src/config.py) 
# availablle to shell scripts

import config

# Print only models that have a defined name
if config.defaultModel:
    print(config.defaultModel)
if config.defaultModelB:
    print(config.defaultModelB)
if getattr(config, 'model_c', None) and isinstance(config.model_c, dict) and config.model_c.get("name"):
    print(config.model_c["name"])
if getattr(config, 'model_d', None) and isinstance(config.model_d, dict) and config.model_d.get("name"):
    print(config.model_d["name"])


