# Changelog

### v2.0 - xxYYYr2025
- major reorg of src directory as follows:
     - moved common code such as config, model helpers, utils to common/
     - moved mcq generation, answer, scoring, etc. workflow (original workflow) to mcq\_workflow/
     - moved new knowledge (nugget) extraction workflow - new - to nugget\_workflow/
     - moved tuning scripts into src/tuning
- Implemented a modules approach with the understading that users would run all scripts
  from the repo root directory, invoking the scripts with "python -m modulename.scriptname"
  for the script at "src/modulename/scriptname.py"
     - internal changes such as changing from "import config" to "import common.config"
     - various other details re paths, etc.
- review\_status.py
     - modified to locate and read JSONL files rather than JSON as
       the \_RESULTS it reviews are JSONL (and have jsonl filename extensions).
- generate\_mcqs
    - add more robust handling of impudent models refusing to follow instructions
      and answer in proper JSON format.
- model\_access
    - implement argo model type (as with alcf, must be at lab or on vpn)
- generate\_answers
    - reads JSONL _and_ JSON just to be more resilient
- generate\_nuggets - new script to create "nuggets" from papers.
    - nugget is the augmented chunk from step 1 in generate\_mcqs, specifically
      formatted in JSON as an ID (DOI ideally) and the augmented chunk (bullet
      point summary and 80-100 word comment).
    - extracts metadata (title, author, ArXiv info if available) from first chunk
    - uses ArXiv ID, DOI, or if not present, looks up DOI to use as a tag to each nugget
    - if cannot find DOI, uses titlewithoutbreaks-firstauthorlastname as the ID
    - uses abstract as the first nugget
    - deduplicate as needed
- config.yml
    - added prompts subsection for generate\_nuggets
- config.py
    - added code to extract generate\_nuggets prompts

### v1.5 - 09Apr2025 (CeC)
- generate\_mcqs
    - write all MCQs out after finishing each file, not by chunk
    - Improve ^C interrupt handling for things like model time outs
    - Write to a debug file when JSON parsing fails in case we want to investigate
- model\_access
    - implement a "test" model to validate the workflow (including generate\_mcqs,
      generate\_answers, and score\_answers) with a stub model both for speed
      (local models are not so fast) and situations where no models are available
     (like working offline)
    - timeout at 60s was hard-coded here - it's now a config variable in config.yml
- generate\_answers
    - reads JSONL to match format of MCQS json files.
- test\_model.py
    - a stub model for testing offline and without the delays associated with
      actual model interactions.
- src/README.md
    - Documents the test infrastructure
- src/test\_workflow.sh
    - Runs through the workflow using just two models (A and B in config.yml)
- src/test\_model\_verification.py
    - Tests to ensure that the stub model works with each component in the workflow.

### v1.4 - 26Mar2025 (CeC)
- Implemented a global "bail out" flag and checks before iniating new threads so that
  one can abort without having to hammer ^C multiple times, and so that fatal errors
  like model access fails can cause a shut-down withous user intervention.

### v1.3 - 24Mar2025 (CeC)
- Add option to run\_workflow.sh to select subset of MCQs. 
- Improved some of the error messages to give the user more useful information.

### v1.2 - 21Mar2025 (CeC)
- Major restructure of generate\_mcqs.py to (a) use two pbars to track both
  chunks processed and successful chunks, (b) warn the user if success rate
  drops below 50% (for whatever reason), and (c) parallelize by chunk rather than
  file, to achieve performance improvement with small number of files.

### v1.1 - 20Mar2025 (CeC)
- New shell (zsh) script that executes the entire workflow. The individual
  python scripts are already paralell in their interactions with the models,
  and the run\_workflow.py script runs those concurrently in the background,
  (i.e., in parallel) as each instances is using a different model.

### v1.0 - 12Mar2025 (CeC)
- Moved all scripts to parallel versions; parallel\_foobar.py now foobar.py
  and legacy scripts are renamed serial\_foobar.py
- implemented pbar, -v, and -q logic for simple\_parse

### v0.b - 12Mar2025 (CeC)
- moved logic to decide on -v, -q, or pbar into config.yml
- created parallel version of score\_answers

### v0.a - 10Mar2025 (CeC)
- fix a json vs jsonl issue that crept up...
- yeah, I'm using hex because I'm not ready to commit to a 1.0!

### v0.9 - 10Mar2025 (CeC)
- created an initial parallel version of generate\_answers.  Works well but 
  (as with parallel\generate\_mcqs, keyboard 
  interrupt (^C) to abort is not so clean (a cosmetic issue for later fixing).

### v0.8 - 09Mar2025 (CeC)
- created an initial parallel version of generate\_mcqs.  Works well but keyboard 
  interrupt (^C) to abort is not so clean (a cosmetic issue for later fixing).

### v0.7 - 08Mar2025 (CeC)
- moved NoOpTdm to config.py for re-use by scripts w/o dupe
- implemented -v -q and pbar in score\_answers
- moved score\_answers prompts into config.yml
- can specify models A and B in config.yml to run the entire workflow withoug
  spacifying models on the command line (making way for easy workflow scripts)
- updated README.md to nudge users to specify models in config.yml vs 
  command line.

### v0.6 - 06Mar2025 (CeC)
- moved additional values (e.g., Temperature) out of code and into config.yml
- Implemented improvements from generate\_mcqs into generate\_answers, including
  -v and -q as well as default (!v and !q) progress bar.
- Changed generate\_answers to write output after each loop rather than saving all
  results and writing the file at the end.
- undid an earlier change that tried to authenticate with ALCF endpoints even if
  you were not using an ALCF-hosted model.
- reorganized repo to put .py scripts in src directory - adjusted the various
  pathnames in scripts and README.md

### v0.5 - 28Feb2025 (CeC)
- changed strategy on logging in generate\_mcqs.py.  Default is now progress bar
  (no logger.info chatter). -q or --quiet is now totally silent unless critical 
  errors are thrown.  -v or --verbose to get progress messages (logger.info) for
  debugging prompts, etc.

### v0.4 - 24Feb2025 (CeC)
- improved consistency of valid JSON creation (via more pointed prompts)
- report stats for each file - MCQ created and misfires (generally invalid
  JSON), i.e., success or failure generating an MCQ for each chunk.

### v0.3 - 21Feb2025 (CeC)
- replaced print statements with logging
- implemented tqdm progress bar (including a null stub to suppress when in default
  INFO logging level which logs what used to be printed by default to monitor progress
- new -q --quiet option to only display progress bar, no INFO meessage (but still will
  log (print) warnings)
- fixed a few string ops by forcing str(var) (just a bit cleaner output, since
  non-string items throw exceptions at string operations like .lower or .strip)

### v0.2 - 11Feb2025 (CeC)
- added jq to environment for easy reading json
- added step to check what models are running prior to firing off generate\_mcqs
- verified (on MacOS CLI) that the entire workflow example, steps 1-8, works
  though there are various errors to be expected (with imperfect data).

### v0.1 - 10Feb2025 (CeC)
- README.md - overhaul initial steps of workflow
- alcf\_inference\_utilities.py - Modified to exit with simple error message in cases where
  no network path available (such as not being local to ALCF or on VPN), avoiding
  100 lines of traceback.
- model\_access.py - Added code to centralize several keys shared across multiple .py scripts,
  including ALCF\_ACCESS\_TOKEN and ALCF\_CHAT\_MODELS.
- environment.yml - Added to enable creation of conda env with all dependencies
  (many were missing and none were documented)
- generate\_mcqs\_config.yaml - Added in contemplation of easier modification of things like
  prompts, etc. but for now on hold for higher priorities.
- requirements.txt - Added to make setup easier, but deprecated in favor of using environment.yml
  which is a more comprehensive snapshot to recreate the conda environment.

### v0 - Original code
