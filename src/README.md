# Structure of this Repository

As illustrated in the main README and workflow scripts (e.g.,
['run\_mcq\_workflow'](https://github.com/auroraGPT-ANL/MCQ-and-SFT-code/blob/main/run_mcq_workflow.sh)),
the code base uses Python modules.  

## Using Modules

1. Two primary differences are introduced moving when transitioning from stand-alone Python scripts to Python modules:

- Instead of calling a script by its file path, you now run it using the 
  '-m' flag with its module name (without the '.py' extension).
  - Example:
Instead of:
```bash
python src/mcq_workflow/generate_mcqs.py
```
you now run:
```bash
python -m mcq_workflow.generate_mcqs
```

2. When importing modules within your package, you now use fully qualified names to indicate
   their location in the package hierarchy.

- Example:
Instead of writing:
```bash
import config
```
you should now write:
```bash
import common.config
```
of, if you prefer to import and alias it:
```bash
from common import config
```

## Modules in this Repository

### Common

This module includes common scripts (used by all three of the modules below) for accessing models,
reading *config.yml*, etc. 

### mcq\_workflow

Scripts in this module implement the 
[MCQ workflow](https://github.com/auroraGPT-ANL/MCQ-and-SFT-code/blob/main/MCQ-Workflow.png).

### nugget\_workflow

Scripts in this module, under construction, aim to implement a new workflow, similar to 
MCQ\_workflow, but extracting unique *nuggets* representing facts.  The workflow will be built out
to query target models to determine if a given nugget is *new* to that model.  This will facilitate
narrowing down the nuggets extracted from the input corpus (*\_PAPERS*) to only those that the
model does not already know, making tuning more efficient.


### tune\_workflow

Scripts in this module implement model tuning, and have yet to be updated and tested.
