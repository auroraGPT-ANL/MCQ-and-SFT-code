# Changelog

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
