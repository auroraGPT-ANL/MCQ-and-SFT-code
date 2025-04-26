# Plan for TPC Spring Hackathon Data Pipelines Team

*May 6-8, Helsinki, Finland*

This hackathon will deepen and build on the code in this repo, which is a workflow that generates
training data in the form of Multiple Choice Question and Answers (MCQA) from scientific papers.

## Background

The code in this repo has been tested on a variety of inference endpoints, including:
* *OpenAi* 
* *Local* (e.g., using LM-studio)
* *ALCF* Inference Endpoints (requires Argonne account)
* *Argo* (requires Argonne account)

There are additional classes that have not been thoroughly tested, including *pb* (submit batch job to Argonne systems; requires Argonne account)

For the Helsinki hackathon we will have access to the following resources, with pre-arranged access 
using your email address:
* Lumi
* A cerebras system provided by Cerebras
* A 50 GPU (A100) cluster provided by NVIDIA

We will want to get this workflow up and running on Lumi and the NVIDIA cluster, as well as adding
the inference endpoints for using models hosted on each of these resources.

## Hackathon Desired Outcomes

Our goal for this hackathon is to take a lightly-tested workflow and harden it for its original purpose
(create high-quality training data from scientific papers), and extend it in two directions:
* Harden the MCQA workflow for others to use.
* Create an alternate approach to MCQA where we extract "new knowldge" from the papers.
* Discuss what a prototype agentic system would look like for this workflow.  For instance, a
system that can execute the workflow periodically for a user, based on prompts
provided by the user, and ideally discovering useful knowledge that the user may not have anticipated including
in the original prompt.

## Potential Task Groups

### Harden (and complete) Current Workflow

The hackathon brings multiple *first-time* users who will likely
identify (and fix) any difficulties, unexpected failure modes, documentation errors, etc.
Additionally, there are a number of "to-do" items that will help to harden the workflow.

The pipeline is functioning up until the collection of scored answers, but the next
action item for the team to address is to add the tuning stage - that is, to use the MCQA
data to fine-tune a target model.  This final  stage of the pipeline 
is coded but has not been tested and integrated with the prior steps in the pipeline.

The following is an unordered list of "to-do" itema that focus on the current workflow 
[(flowchart)](https://github.com/auroraGPT-ANL/MCQ-and-SFT-code/blob/CeC/MCQ-Workflow.png).

* Integrate the fine-tuning scripts with the existing workflow (which is tested only up to,
but not including, the fine-tuning stage.  All steps up to that point can run on any host
(including a laptop) as the heavy lifting is done by remote models accessed via inferance endpoints.
The fine-tuning step requires significantly more horsepower than a laptop, thus we envision two
strategies to be implemented:
  * A functioning workflow running on a server with sufficient capacity (and GPUs), and, alternatively
  * A workflow that runs on a laptop but submits a job to an HPC system to perform the fine-tuning step.
* Other more pedestrian needs include:
  * Address clean shutdown of the multi-threaded code on detection of keyboard interrupt.
  * (list to be expanded)
* Add  model types (and endpoints) for new resources starting with those made avaialble for the hackathon.


### New Workflow Development

The current MCQ workflow (src/mcq\_workflow) creates and refines (answers/scores) MCQ's
and answers, i.e., MCQAs, from scientific papers, for use in fine-tuning models.  We would like to create an alternate 
workflow (src/nuggets\_workflow) that:
1. extracts knowlege *nuggets* (statements of facts or findings) from the papers, then
2. probes a target model to determine which nuggets are new information


### Transition from Static Workflow to an Agentic, semi-Autonomous System

The hackathon team will discuss strategies for implementing an agentic system based on these workflows
(ideally tne new-nuggets workflow).  A useful discussion on this topic, which should help to drive the
strategic discussion can be found in this 
[post at the LangChain blog](https://blog.langchain.dev/how-to-think-about-agent-frameworks/).

In this repo there is a branch that contains an incomplete agent implementation, intended as a simple
exercise.  The team should discuss how we might create a semi-autonomous system that would 
(with initial prompting to deccibe the information of interest) continually
find new information (papers, articles, etc.), create high-quality training data from this information,
and fine-tune a target model for the user.  The existing experiment for this can be either leveraged or
ignored, at the group's discretion.

