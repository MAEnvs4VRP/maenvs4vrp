from maenvs4vrp.learning.mardam.baselines._base import Baseline

import torch
import torch.nn as nn


class CriticBaseline(Baseline):
    def __init__(self, learner, cust_count, use_qval=False, use_cumul_reward=False):
        super().__init__(learner, use_cumul_reward)
        self.use_qval = use_qval
        self.project = nn.Linear(
            cust_count + 1, cust_count + 1 if use_qval else 1, bias=False
        )

    def eval_step(self, cur_veh_mask, learner_compat, cust_idx):
        compat = learner_compat.clone()
        compat.masked_fill_(cur_veh_mask == True, 0)
        val = self.project(compat)
        if self.use_qval:
            val = val.gather(2, cust_idx.unsqueeze(1).expand(-1, 1, -1))
        return val.squeeze(1)

    def __call__(self, env, td):
        node_stat_obs = td["observations"]["node_static_obs"]

        actions, logps, rewards, bl_vals, step_mask = [], [], [], [], []

        self.learner._encode_customers(node_stat_obs)

        while not td["done"].all():
            # rollover the observations
            node_dyn_obs = td["observations"]["node_dynamic_obs"]
            action_mask = td["observations"]["action_mask"]
            # self_obs = td['observations']['agent_obs']
            # global_obs = td['observations']['global_obs']
            agents_mask = td["observations"]["agents_mask"]
            agents_obs = td["observations"]["other_agents_obs"]
            node_obs = torch.cat((node_stat_obs, node_dyn_obs), dim=2)

            cur_veh_idx = td["cur_agent_idx"]
            mask = ~env.td_state["agents"]["feasible_nodes"].clone()

            veh_repr = self.learner._repr_vehicle(agents_obs, cur_veh_idx, mask)
            compat = self.learner._score_customers(veh_repr)
            logp = self.learner._get_logp(compat, ~action_mask.unsqueeze(1))

            cust_idx = logp.exp().multinomial(1)
            if not (self.use_cumul and bl_vals):
                bl_vals.append(
                    self.eval_step(~action_mask.unsqueeze(1), compat, cust_idx)
                )

            actions.append((cur_veh_idx, cust_idx))
            logps.append(logp.gather(1, cust_idx))

            td["action"] = cust_idx
            td = env.step(td)
            rewards.append(td["reward"] + td["penalty"])
            step_mask.append(~td["done"].unsqueeze(1))

        if self.use_cumul:
            rewards = torch.stack(rewards).sum(dim=0)
            bl_vals = bl_vals[0]
        return actions, logps, rewards, bl_vals, step_mask

    def parameters(self):
        return self.project.parameters()

    def state_dict(self):
        return self.project.state_dict()

    def load_state_dict(self, state_dict):
        return self.project.load_state_dict(state_dict)

    def to(self, device):
        self.project.to(device=device)
