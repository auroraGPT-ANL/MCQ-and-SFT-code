# Structure of this Repository

As illustrated in the main README and workflow scripts (e.g.,
['run\_mcq\_workflow'](https://github.com/auroraGPT-ANL/MCQ-and-SFT-code/blob/main/run_mcq_workflow.sh)),
the code base uses Python modules.  

## Modules

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
