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
* A 50-GPU (A100) cluster provided by NVIDIA

We will want to get this workflow up and running on Lumi and the NVIDIA cluster, as well as adding
the inference endpoints for using models hosted on each of these resources.

## Goals for this Hackathon

Over the 2.5-day Sauna hackathon, our team will strengthen multiple related workflows (pipelines) for generating
training data from scientific papers. We have at least four work threads to select from, though we  may not be able
to tackle all of them.  These are: 
1. Harden the existing MCQ workflow, 
2. Expand a new approoach -- knowledge-nugget (NKN) workflow, 
3. Update and harden the model fine-tuning step that follows the MCQ and NKN workflows, and
4. Explore what an agentic design of the MCQ workflow would look like, and possibly design and begin
implementing a minimal prototype of the underlying agentic framework.

As described in this repo, the MCQ workflow converts PDFs to JSON, generates multiple-choice questions and
answers, and has models score each other’s answers.
The Nuggets (NKN) workflow – still under development – will extract factual “knowledge nuggets” from
papers and identify which are new to the model. The fine-tuning stage applies tools for fine-tuning a
model on these outputs and the code for this tage is present but unfinished.

## Organization and Process

We will have 6-8 people, so we don't need a rigid organization of sub-groups, and can form pairs or small teams
at our discretion throughout the hackathon. There are no fixed assignments – once we have an overall plan of
action targeting one or more of the above four threads, we will develop a high-level set of to-do items,
reviewing and updating this list after each break during the hackathon.  All groups draw tasks from the
resulting  shared to-do list and can re-distribute work as needed. For example, one group might initially
focus on MCQ bugs while another tackles nuggets, but members are encouraged to switch areas or collaborate
across groups. 

We will share the *sauna-hack* branch.  The *main* branch is locked down to we don't accidentally merge
to it. Because some of our action items involve *hardening* the code, we expect to periodically merge our
changes via pull request (PR).


## 0. Overview and kickoff
- **Day 0 (30 min)** – welcome, repo walk‑through, review of this plan  
- **Agentic‑framework discussion (30 – 45 min)** – brainstorm minimal “agent” wrapper that can:  
  1. call the existing MCQ workflow end‑to‑end,  
  2. expose a plug‑in point so additional workflows (NKN, others) can drop in with the same interface.  
- **Decision point** – by consensus, decide whether one or two groups will focus on an agentic prototype
track while the remaining group(s) work on the core MCQ / NKN / Tune tasks.

> _If no group elects the agentic track, all groups use the Task List in §4. If one or two groups do,
they follow the Agentic tasks in §5._

## 1. Goals
Deliver, in 2.5 days:
- Harden **MCQ** pipeline  
- Expand **NKN** (nuggets) pipeline  
- Implement **Tuning** pipeline that consumes MCQ/NKN data  
- *(Optional)* Bootstrap an **Agentic framework** able to orchestrate MCQ now and NKN later  

## 2. Organization

The hackathon is a set of 90-minute sprints with breaks or meals in between.

| Role | Description |
|------|-------------|
| **Groups A–C** | 3 groups, 2–3 people each. After kickoff, choose tracks: Core (§4) or Agentic (§5). |
| **Session syncs** | Session start: 3‑min stand‑ups and wrap‑ups |
| **Task board** | GitHub Projects **“Sauna‑Hack”** – tasks labelled `MCQ`, `NKN`, `TUNE`, `AGENT` |
| **Code rules** | Work in `sauna-hack` or feature branches → PR → 1 owner review (CODEOWNERS) |

## 3. Repo & Branches

There are multiple branches in this repo and for our work this week we will use the sauna-hack branch. If
we elect to pursue an initial agentic system we will create a new branch for that.  There are two branches
(stable-legacy and agent-system) that can be ignored for this hackathon.

```
main         ← production (stable/protected- may not merge until last session)
sauna-hack   ← shared hack branch (open write)
feature-*    ← individual branches for PRs
agentic/...  ← optional agentic track base branch (if created)
```

## 4. Core Task List (MCQ / NKN / Tuning)

### MCQ Workflow Hardening
- Update src/common/model\_access. py with new model classes and endpoints for the hackathon resources to be used.
- Integrate & test fine-tuning step  
- Training strategies (GPU server + HPC batch)  
- Model backend expansion  
- Config & secrets templates  
- End‑to‑end tests & documentation  

### NKN Workflow Expansion
- Refine factoid extraction  
- Implement novelty filter  
- Assemble full nuggets pipeline  
- Validate on sample papers  
- Documentation  

### Tuning Pipeline Setup
- Data preparation from MCQ/NKN  
- Fine‑tuning experiment (small‑scale)  
- HPC execution path  
- Integration into scripts  
- Documentation  

## 5. Agentic Prototype Track (optional)
If chosen, Group B (or B+C) aims for a minimal **Agent Runner** that:
1. Accepts a list of paper PDFs.  
2. Spawns an **MCQ Agent** that wraps existing `run_mcq_workflow.sh` (or Python orchestrator).  
3. Stores outputs in a standardized `/artifacts/{workflow}/{paper_id}/…` layout.  
4. Provides a plug‑in registry so future agents (e.g., **NKN Agent**) can drop in with the same method signature `process(paper_meta) -> artifacts_path`.

### Agentic tasks

An agentic design must be fleshed out, at least in terms of general architecture and the selection of any underlying frameworks
or tools.  Below is an example set of tasks that will need to be refined.

| Priority | Task |
|----------|------|
| P0 | Define agent interface (Python `ABC` or dataclass) and artifact spec |
| P1 | Implement MCQ agent wrapper (shell out or import) |
| P1 | Demonstrate running two papers concurrently (threads or asyncio) |
| P2 | Draft YAML/JSON config that lists which agents to run per paper |
| P2 | Skeleton NKN agent module calling current nugget extractor |
| P3 | CLI: `python agent_runner.py --config config.yml` |
| P3 | README for agent framework & future extension guide |

*Stretch:* minimal scheduler (queue + worker pool) and simple web dashboard.

---

### Deliverables
1. MCQ pipeline passes end‑to‑end integration test.  
2. Nuggets extraction produces validated JSON on sample set.  
3. Tuning script trains a toy model (or submits job) without error.  
4. *If agentic track pursued:* Agent Runner runs MCQ agent on at least two papers and generates artifacts.
