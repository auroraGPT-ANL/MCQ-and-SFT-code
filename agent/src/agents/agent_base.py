# agent/src/agents/agent_base.py

from abc import ABC, abstractmethod

class Agent(ABC):
    """
    All pipeline agents will inherit from this.
    Each agent takes a 'context' dict as input, does its work,
    and returns a dict of outputs to merge back into that context.
    """

    @abstractmethod
    def run(self, context: dict) -> dict:
        """
        Perform the agent’s task.
        
        Args:
          context: a dict carrying inputs (e.g. project_root, p_value, parsed_dir, etc.)
        
        Returns:
          A dict of new context entries (e.g. {"parsed_dir": "/path/to/_JSON"}).
        """
        pass

