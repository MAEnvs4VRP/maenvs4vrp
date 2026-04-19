import torch
from maenvs4vrp.core.env_agent_reward import RewardFn


class DenseReward(RewardFn):
    """
    PCVRPTW dense reward class.
    """

    def __init__(self):
        """
        Constructor.

        Args:
            n/a.

        Returns:
            None.
        """
        self.env = None

    def set_env(self, env):
        """
        Set environment.

        Args:
            env(AECEnv): Environment.

        Returns:
            None.
        """
        self.env = env

    def get_reward(self, action):
        """
        Get reward and penalty.

        Args:
            action(torch.Tensor): Tensor with agent moves.

        Returns:
            reward(torch.Tensor): Reward.
            penalty(torch.Tensor): Penalty.
        """

        reward = (
            -self.env.td_state["cur_agent"]["cur_ttime"].clone()
            + self.env.td_state["profits"].gather(1, action).clone()
        )
        penalty = torch.zeros_like(action, dtype=torch.float, device=self.env.device)

        return reward, penalty


class SparseReward(RewardFn):
    """
    PCVRPTW sparse reward class.
    """

    def __init__(self):
        """
        Constructor.

        Args:
            n/a.

        Returns:
            None.
        """
        self.env = None

    def set_env(self, env):
        """
        Set environment.

        Args:
            env(Environment): Environment.

        Returns:
            None.
        """
        self.env = env

    def get_reward(self, action):
        """
        Get reward and penalty.

        Args:
            action(torch.Tensor): Tensor with agent moves.

        Returns:
            reward(torch.Tensor): Reward.
            penalty(torch.Tensor): Penalty.
        """

        reward = torch.zeros_like(action, dtype=torch.float, device=self.env.device)
        penalty = torch.zeros_like(action, dtype=torch.float, device=self.env.device)

        # compute penalty if env has unvisited nodes
        is_last_step = self.env.td_state["is_last_step"]
        final_reward = -self.env.td_state["agents"]["cum_ttime"].sum(
            1, keepdim=True
        ) + self.env.td_state["agents"]["cum_profit"].sum(-1, keepdim=True)
        reward[is_last_step] = final_reward[is_last_step]
        return reward, penalty
