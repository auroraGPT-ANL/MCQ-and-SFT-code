#!/usr/bin/env python
import common.config

# First try your workflow section…
workflow_cfg = getattr(common.config, 'workflow', {})
contestants = workflow_cfg.get('contestants', [])

if contestants:
    # print each model named in config.local.yml → workflow.contestants
    for model in contestants:
        if model:
            print(model)
else:
    # fallback to the old-style defaults
    if common.config.defaultModel:
        print(common.config.defaultModel)
    if common.config.defaultModelB:
        print(common.config.defaultModelB)
    if getattr(common.config, 'model_c', None) \
       and isinstance(common.config.model_c, dict) \
       and common.config.model_c.get("name"):
        print(common.config.model_c["name"])
    if getattr(common.config, 'model_d', None) \
       and isinstance(common.config.model_d, dict) \
       and common.config.model_d.get("name"):
        print(common.config.model_d["name"])

