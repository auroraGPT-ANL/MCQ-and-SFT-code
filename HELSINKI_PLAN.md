## Plan for TPC Spring Hackathon Data Pipelines Team

This hackathon will deepen and build on the code in this repo, which is a workflow that generates
training data in the form of Multiple Choice Question and Answers (MCQA) from scientific papers.

### Background

The code in this repo has been tested on a variety of inference endpoints, including:
* ALCF Inference Endpoints (requires Argonne account)
* Argo (requires Argonne account)
* Local (e.g., using LM-studio)
* OpenAi 

There are additional classes that have not been thoroughly tested, including:
* pb (submit batch job to Argonne systems; requires Argonne account)
* cafe (a server run by a colleague, but has not been tested)

For the Helsinki hackathon we will have access to the following resources, with pre-arranged access 
using your email address:
* Lumi
* A cerebras system provided by Cerebras
* A 50 GPU (A100) cluster provided by NVIDIA

An orientation session is scheduled for 29-April for the NVIDIA cluster and we will have orientation for
the other two resources on the first morning of the hackathon.

### Hackathon Ooutcomes

Our goal for this hackathon is to take a lightly-tested workflow and harden it for its original purpose
(create high-quality training data from scientific papers), and extend it in two directions:
* Create an alternate approach to MCQA where we extract "new knowldge" from the papers.
* Create an agentic system that can execute the workflow periodically for a user, based on prompts
provided by the user.

#### Harden (and complete) Current Workflow

In this first step, the hackathon team brings multiple *first-time* users who will
identify (and fix) any difficulties, unexpected failure modes, documentation errors, etc.
Additionally, there are a number of "to-do" items that will help to harden the workflow.

The pipeline is functioning up until the collection of scored answers, but the next
action item for the team to address is to add the tuning stage - that is, to use the MCQA
data to fine-tune a target model.  This final  stage of the pipeline 
is coded but has not been tested and integrated with the prior steps in the pipeline.

The following is an unordered list of "to-do" itema that focus on the current workflow 
[(flowchart)](https://github.com/auroraGPT-ANL/MCQ-and-SFT-code/blob/CeC/MCQ-Workflow.png).

* Address clean shutdown of the multi-threaded code on detection of keyboard interrupt.
* (list to be expanded)


#### New Workflow Development

The current MCQ workflow (src/mcq\_workflow) creates and refines (answers/scores) MCQ's
and answers, i.e., MCQAs, from scientific papers, for use in fine-tuning models.  We would like to create an alternate 
workflow (src/nuggets\_workflow) that:
1. extracts knowlege *nuggets* (statements of facts or findings) from the papers, then
2. probes a target model to determine which nuggets are new information


#### Transition from Static Workflow to an Agentic and semi-Autonomous System

In this repo there is a branch that contains an incomplete agent implementation, intended as a simple
exercise.  The team should discuss how we might create a semi-autonomous system that would 
(with initial prompting to deccibe the information of interest) continually
find new information (papers, articles, etc.), create high-quality training data from this information,
and fine-tune a target model for the user.  The existing experiment for this can be either leveraged or
ignored, at the group's discretion.

