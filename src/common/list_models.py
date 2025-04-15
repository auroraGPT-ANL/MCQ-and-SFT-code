#!/usr/bin/env python
# make the list of models (defined in config.yml, extracted by src/common/config.py) 
# availablle to shell scripts

import common.config

# Print only models that have a defined name
if common.config.defaultModel:
    print(common.config.defaultModel)
if common.config.defaultModelB:
    print(common.config.defaultModelB)
if getattr(common.config, 'model_c', None) and isinstance(common.config.model_c, dict) and common.config.model_c.get("name"):
    print(common.config.model_c["name"])
if getattr(common.config, 'model_d', None) and isinstance(common.config.model_d, dict) and common.config.model_d.get("name"):
    print(common.config.model_d["name"])


