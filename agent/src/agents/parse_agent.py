# agent/src/agents/parse_agent.py

from agent_base import Agent
import common.simple_parse as simple_parse

class ParseAgent(Agent):
    def run(self, context: dict) -> dict:
        # 1) call your existing parser
        simple_parse.main()  
        # 2) return the directory where JSON lives
        project_root = context["project_root"]
        return {"parsed_dir": f"{project_root}/_JSON"}

