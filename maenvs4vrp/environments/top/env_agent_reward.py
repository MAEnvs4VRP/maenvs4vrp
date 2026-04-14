import torch
from tensordict import TensorDict
from maenvs4vrp.core.env_agent_reward import RewardFn

from typing import Optional, List


class DenseReward(RewardFn):
    """Reward class.
    """

    def __init__(self):
        """Constructor

        """
        self.env = None

    def set_env(self, env):
        self.env = env

    def get_reward(self, action):
        """
        
        """
        reward = self.env.td_state['profits'].gather(1, action).clone()
        penalty = torch.zeros_like(action, dtype = torch.float, device=self.env.device)

        return reward, penalty



class SparseReward(RewardFn):
    """Reward class.
    """

    def __init__(self):
        """Constructor

        """
        self.env = None

    def set_env(self, env):
        self.env = env

    def get_reward(self, action):
        """
        
        """

        reward = torch.zeros_like(action, dtype = torch.float, device=self.env.device)
        penalty = torch.zeros_like(action, dtype = torch.float, device=self.env.device)

        # compute penalty if env has unvisited nodes 
        is_last_step = self.env.td_state['is_last_step']
        final_reward = self.env.td_state['agents']['cum_profit'].sum(1, keepdim = True)
        reward[is_last_step] = final_reward[is_last_step]
        return reward, penalty