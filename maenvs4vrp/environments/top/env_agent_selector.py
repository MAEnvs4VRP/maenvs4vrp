from maenvs4vrp.core.env import AECEnv
from maenvs4vrp.core.env_agent_selector import BaseSelector
import torch


class RoundRobin(BaseSelector):
    def __init__(self):
        super().__init__()

        """

        """

    def set_env(self, env: AECEnv):
        super().set_env(env)

    def _next_agent(self):
        """Returns the next agent

        Returns:
            Tensor: next agent
        """
        selected_agent = (self.env.td_state["cur_agent_idx"] + 1) % self.env.num_agents
        return selected_agent


class RandomSelector(BaseSelector):
    def __init__(self):
        super().__init__()

        """

        """

    def set_env(self, env: AECEnv):
        super().set_env(env)

    def _next_agent(self):
        """Returns the next agent

        Returns:
            Tensor: next agent
        """
        selected_agent = torch.multinomial(
            self.env.td_state["agents"]["active_agents_mask"].float(), 1
        ).to(self.env.device)
        return selected_agent


class AgentSelector(BaseSelector):
    def __init__(self):
        super().__init__()

        """

        """

    def set_env(self, env: AECEnv):
        super().set_env(env)

    def _next_agent(self):
        """Returns the next agent

        Returns:
            Tensor: next agent
        """
        avail = (
            torch.arange(self.env.num_agents, dtype=torch.float)
            .unsqueeze(0)
            .repeat(*self.env.batch_size, 1)
            .to(self.env.device)
        )
        avail[~self.env.td_state["agents"]["active_agents_mask"]] = float("inf")
        selected_agent = avail.argmin(1, keepdim=True)
        return selected_agent


class SmallestTimeAgentSelector(BaseSelector):
    def __init__(self):
        super().__init__()
        """

        """

    def set_env(self, env: AECEnv):
        super().set_env(env)

    def _next_agent(self):
        """Returns the next agent

        Returns:
            Tensor: next agent
        """
        avail = self.env.td_state["agents"]["cur_time"].clone()
        avail[~self.env.td_state["agents"]["active_agents_mask"]] = float("inf")
        selected_agent = avail.argmin(1, keepdim=True)
        return selected_agent
