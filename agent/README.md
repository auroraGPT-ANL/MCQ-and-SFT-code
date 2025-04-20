## Agentic Flow Version of MCQ_ and Nugget_ Workflows

### Set up

1. Follow the guide in the root README to clone this repo, create data directories, set configuration details (such as choices of models), etc.

2. Add *agent/* to your PYTHONPATH, so your .zshrc file should include this:
```bash
# MCQ-and-SFT-code project
export MCQ_ROOT="$HOME/Dropbox/MyCode/ALCF/MCQ-and-SFT-code"
export PYTHONPATH="$MCQ_ROOT/src:$MCQ_ROOT/agent/src${PYTHONPATH:+:}$PYTHONPATH"
```
After making this change, you can update with
```bash
source ~/.zshrc
```
