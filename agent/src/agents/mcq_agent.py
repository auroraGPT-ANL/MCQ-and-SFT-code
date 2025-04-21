import os
from agents.agent_base import Agent
from mcq_workflow.generate_mcqs import generate_mcqs
from common import config

class MCQAgent(Agent):
    def run(self, ctx):
        parsed_dir = ctx["parsed_dir"]
        mcq_out    = ctx.get("mcq_out", config.mcq_dir)
        os.makedirs(mcq_out, exist_ok=True)
        out = generate_mcqs(input_dir=parsed_dir,
                            threads=ctx["p_value"],
                            verbose=ctx["v_flag"],
                            output_dir=mcq_out,
                            force=ctx["force_mcq"])
        return {"mcq_out": out}

