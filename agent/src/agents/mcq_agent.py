# agent/src/agents/mcq_agent.py

from agent_base import Agent
from mcq_workflow.generate_mcqs import generate_mcqs

class MCQAgent(Agent):
    def run(self, context: dict) -> dict:
        parsed_dir = context["parsed_dir"]
        p_value   = context["p_value"]
        v_flag    = context["v_flag"]
        # call your generate_mcqs entrypoint
        output_file = generate_mcqs(input_dir=parsed_dir,
                                    threads=p_value,
                                    verbose=bool(v_flag))
        return {"mcq_file": output_file}

