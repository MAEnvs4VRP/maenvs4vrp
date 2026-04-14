from maenvs4vrp.core.env import AECEnv
from maenvs4vrp.core.env_agent_selector import BaseSelector
import torch


class AgentSelector(BaseSelector):
    """
    CVRPTW agent selector class.
    """
    def __init__(self):
        super().__init__()

        """
        Constructor.

        Args:
            n/a.

        Returns:
            None.
        """

    def set_env(self, env: AECEnv):
        """
        Set environment.

        Args:
            env(AECEnv): Environment.

        Returns:
            None.
        """
        super().set_env(env)

    def _next_agent(self):
        """
        Return the next agent.

        Args:
            n/a.

        Returns:
            selected_agent(torch.Tensor): Next agent.  
        """
        avail = torch.arange(self.env.num_agents, dtype = torch.float).unsqueeze(0).repeat(*self.env.batch_size, 1).to(self.env.device)
        avail[~self.env.td_state['agents']['active_agents_mask']] = float('inf')
        selected_agent = avail.argmin(1, keepdim = True)
        return selected_agent
    

class RandomSelector(BaseSelector):
    """
    CVRPTW random agent selector class.
    """
    def __init__(self):
        super().__init__()

        """
        Constructor.

        Args:
            n/a.

        Returns:
            None.
        """

    def set_env(self, env: AECEnv):
        super().set_env(env)

        """
        Set environment.

        Args:
            env(AECEnv): Environment.

        Returns:
            None.
        """

    def _next_agent(self):
        """
        Return the next agent.

        Args:
            n/a.

        Returns:
            selected_agent(torch.Tensor): Next agent. 
        """
        selected_agent = torch.multinomial(self.env.td_state['agents']['active_agents_mask'].float(), 1).to(self.env.device)
        return selected_agent
    


class SmallestTimeAgentSelector(BaseSelector):
    """
    CvRPTW smallest time agent selector class.
    """
    def __init__(self):
        super().__init__()
        """
        Constructor.

        Args:
            n/a.

        Returns:
            None.
        """

    def set_env(self, env: AECEnv):
        super().set_env(env)

        """
        Set environment.

        Args:
            env(AECEnv): Environment.

        Returns:
            None.
        """

    def _next_agent(self):
        """
        Return the next agent.

        Args:
            n/a.

        Returns:
            selected_agent(torch.Tensor): Next agent. 
        """
        avail = self.env.td_state['agents']['cur_time'].clone()
        avail[~self.env.td_state['agents']['active_agents_mask']] = float('inf')
        selected_agent = avail.argmin(1, keepdim = True)
        return selected_agent