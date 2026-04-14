import torch
from tensordict import TensorDict

from typing import Optional, Dict, Callable, Union, List

import warnings

from maenvs4vrp.core.env_generator_builder import InstanceBuilder
from maenvs4vrp.core.env_observation_builder import ObservationBuilder
from maenvs4vrp.core.env_agent_selector import BaseSelector
from maenvs4vrp.core.env_agent_reward import RewardFn
from maenvs4vrp.core.env import AECEnv
from maenvs4vrp.utils.ops import gather_by_index, get_distance

class Environment(AECEnv):
    """
    HCVRP environment generator class.

    """
    def __init__(self,
                instance_generator_object: InstanceBuilder,  
                obs_builder_object: ObservationBuilder,
                agent_selector_object: BaseSelector,
                reward_evaluator: RewardFn,
                seed=None,         
                device: Optional[str] = None,
                batch_size: Optional[torch.Size] = None):

        """
        Constructor.

        Args:
            instance_generator_object(InstanceBuilder): Generator instance.
            obs_builder_object(ObservationBuilder): Observations instance.
            agent_selector_object(BaseSelector): Agent selector instance
            reward_evaluator(RewardFn): Reward evaluator instance.
            seed(int): Random number generator seed. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to None.
            batch_size(torch.Size): Batch size. Defaults to None.
        """

        self.version = 'v0'
        self.env_name = 'hcvrp'

        # seed the environment
        if seed is None:
            self._set_seed(self.DEFAULT_SEED)
        else:
            self._set_seed(seed)

        self.agent_selector = agent_selector_object
        self.inst_generator = instance_generator_object
        self.inst_generator._set_seed(self.seed)
        self.obs_builder = obs_builder_object
        self.obs_builder.set_env(self)
        self.reward_evaluator = reward_evaluator
        self.reward_evaluator.set_env(self)   
        self.env_nsteps = 0

        if device is None:
            self.device = self.inst_generator.device
        else:
            self.device = device
            self.inst_generator.device = device

        if batch_size is None:
            self.batch_size =  self.inst_generator.batch_size
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)
            self.inst_generator.batch_size = torch.Size(batch_size)
            
        self.td_state = TensorDict({}, batch_size=self.batch_size, device=self.device)

        
    def observe(self, td: TensorDict, obs_list=None)-> TensorDict:
        """
        Retrieve agent environment observations.

        Args:
            is_reset(bool): If the environment is on reset. Defauts to False.

        Returns
            td_observations(TensorDict): Current agent observaions and masks dictionary.
        """

        td_observations = self.obs_builder.get_observations(obs_list=obs_list)

        if obs_list is not None and 'action_mask' in obs_list:
            self._update_curr_agent_feasibility()
            td_observations['action_mask'] = self.td_state['cur_agent']['action_mask'].clone()
        if obs_list is not None and 'active_agents_mask' in obs_list:
            td_observations['active_agents_mask'] = self.td_state['agents']['active_agents_mask'].clone()
        if obs_list is not None and 'agents_action_mask' in obs_list:
            self._update_all_agents_feasibility()
            td_observations['agents_action_mask'] = self.td_state['agents']['action_mask'].clone()
        if obs_list is not None and 'agent_cur_node_idx' in obs_list:
            td_observations['agent_cur_node_idx'] = self.td_state['cur_agent']['cur_node_idx'].clone()
        if obs_list is not None and 'agents_cur_node_idx' in obs_list:
            td_observations['agents_cur_nodes_idx'] = self.td_state['agents']['cur_node_idx'].clone()

        td['observations'] = td_observations
        return td

    
    def sample_action(self, td: TensorDict, action_without_agent=False)-> TensorDict:
        """
        Compute a random action from available actions to current agent.

        Args:
            td(TensorDict): Environment instance tensor.

        Returns:
            td(TensorDict): Environment instance tensor with updated action.
        """
        if action_without_agent:
            feasible_nodes = self.td_state['agents']['action_mask'].any(axis=1)
            action = torch.multinomial(feasible_nodes.float(), 1).to(self.device)
        else:
            if 'next_agent' in td:
                cur_agent_idx = td['next_agent']
                action_mask = self.td_state['agents']['action_mask'].gather(1, cur_agent_idx[:,:,None].expand(-1, -1, self.num_nodes)).squeeze(1).clone()
                action = torch.multinomial(action_mask.float(), 1).to(self.device)
            else:
                action = torch.multinomial(self.td_state['cur_agent']["action_mask"].float(), 1).to(self.device)
        td['next_action'] = action
        return td

    def sample_agent(self, td: TensorDict, agent_given_action=False)-> TensorDict:
        """
        Compute a random agent from available agents.

        Args:
            td(TensorDict): Environment instance tensor.
            agent_given_action(bool, optional): If True, sample an agent given the action. Defaults to False.

        Returns:
            td(TensorDict): Environment instance tensor with updated agent.
        """
        if agent_given_action:
            action = td['next_action']
            # ensure action is shape [B, 1]
            if action.dim() == 1:
                action = action.unsqueeze(-1)

            # agents.action_mask: [B, num_agents, N]
            # gather mask for the chosen action -> [B, num_agents, 1] -> squeeze -> [B, num_agents]
            idx = action.unsqueeze(1).expand(-1, self.num_agents, -1)   # [B, num_agents, 1]
            feasible_agents_mask = self.td_state['agents']['action_mask'].gather(2, idx).squeeze(-1)

            # also require agent to be active
            feasible_agents_mask = feasible_agents_mask & self.td_state['agents']['active_agents_mask']

            # Force agent 0 when no agent is feasible for a batch entry.
            # Sample only for batch rows that have at least one feasible agent.
            has_feasible = feasible_agents_mask.any(dim=1)  # [B]
            B = feasible_agents_mask.size(0)
            agent = torch.zeros((B, 1), dtype=torch.int64, device=self.device)  # default agent 0

            if has_feasible.any():
                feasible_rows = torch.nonzero(has_feasible, as_tuple=True)[0]
                sampled = torch.multinomial(feasible_agents_mask[has_feasible].float(), 1).to(self.device)
                agent[feasible_rows] = sampled

        else:
            agent = torch.multinomial(self.td_state['agents']['active_agents_mask'].float(), 1).to(self.device)
        td['next_agent'] = agent
        return td

    def sample_joint(self, td: TensorDict) -> TensorDict:
        """
        Sample both agent and action simultaneously from the joint feasible space.
        
        Args:
            td(TensorDict): Environment instance tensor.

        Returns:
            td(TensorDict): Environment instance tensor with updated agent and action.
        """
        num_nodes = self.num_nodes
        
        # Get action mask for each agent
        action_mask = self.td_state['agents']['action_mask']  # [B, num_agents, N]
        joint_mask = action_mask.reshape(*self.batch_size, -1)  # [B, num_agents * N]

        joint_indices = torch.multinomial(joint_mask.float(), 1).squeeze(-1)  
        
        # Decode joint index to (agent_idx, action_idx)
        agent = (joint_indices // num_nodes).unsqueeze(-1)  # [B, 1]
        action = (joint_indices % num_nodes).unsqueeze(-1)  # [B, 1]

        td['next_agent'] = agent
        td['next_action'] = action
        return td    

    def reset(self, 
                num_agents: int = None,
                num_nodes: int = None,
                min_nodes: float = None,
                max_nodes: float = None,
                min_capacity: float = None,
                max_capacity: float = None,
                min_demand: int = None,
                max_demand: int = None,
                min_speed: float = None,
                max_speed: float = None,
                instance_name: str | None = None,
                sample_type: str = 'random',
                instance_dict: Dict | None = None,
                force_visit: bool = False,
                batch_size: Optional[torch.Size] = None,
                n_augment: Optional[int] = None,
                seed: int | None = None,
                device: Optional[str] = "cpu") -> TensorDict:
        """
        Reset the environment.

        Args:
            num_agents (int | None): Number of vehicles/agents to generate. Uses generator default if None.
            num_nodes (int | None): Number of nodes (including depot). Uses generator default if None.
            min_nodes, max_nodes: Range for node counts when sampling (optional).
            min_capacity, max_capacity (float | None): Range for vehicle capacities.
            min_demand, max_demand (int | None): Range for node demands.
            min_speed, max_speed (float | None): Range for vehicle speeds.
            instance_name (str | None): Name of a saved instance to load (if supported).
            sample_type (str): Sampling mode: 'random', 'augment', or 'saved'. Default 'random'.
            instance_dict (Dict | None): If provided, use this instance instead of sampling.
            force_visit (bool): If True, force agents to visit all feasible customer nodes before finishing.
            batch_size (int | torch.Size | None): Batch shape or integer batch size. Integer will be converted to torch.Size.
            n_augment (int | None): Number of augmentations when sampling (used for augment modes).
            seed (int | None): RNG seed for reproducibility.
            device (str | None): Compute device ('cpu' or 'cuda'). Defaults to "cpu" or uses the instance generator default.

        Returns:
            TensorDict: The initialized environment state TensorDict containing observations, agents, nodes and metadata.
        """

        if seed is not None:
            self._set_seed(seed)

        if batch_size is None:
            batch_size = self.batch_size 
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)
            self.inst_generator.batch_size = torch.Size(batch_size)

        if force_visit is not None:
            self.force_visit = force_visit

        if instance_dict:
            instance_info = instance_dict
        else:
            instance_info = self.inst_generator.sample_instance(num_agents=num_agents, 
                                                            num_nodes=num_nodes, 
                                                            min_nodes=min_nodes,
                                                            max_nodes=max_nodes,
                                                            min_demand=min_demand,
                                                            max_demand=max_demand,
                                                            min_capacity=min_capacity,
                                                            max_capacity=max_capacity,
                                                            min_speed=min_speed,
                                                            max_speed=max_speed,
                                                            instance_name=instance_name,
                                                            sample_type=sample_type,
                                                            batch_size=batch_size,
                                                            n_augment=n_augment,
                                                            seed=seed,
                                                            device=device)

        self.num_nodes = instance_info['num_nodes']
        self.num_agents = instance_info['num_agents']

        if 'n_digits' in instance_info:
            self.n_digits = instance_info['n_digits'] 
        else:
            self.n_digits = None

        self.td_state = instance_info['data']

        self.td_state['done'] = torch.zeros(*batch_size, dtype=torch.bool)
        self.td_state['is_last_step'] = torch.zeros(*batch_size, dtype=torch.bool)
        self.td_state['depot_loc'] = self.td_state['coords'].gather(1, self.td_state['depot_idx'][:,:,None].expand(-1, -1, 2))

        self.td_state['nodes'] = TensorDict(
                                    source={'cur_demands': self.td_state['demands'].clone(),
                                            'active_nodes_mask': torch.ones((*batch_size, self.num_nodes),dtype=torch.bool, device=self.device)},
                                    batch_size=batch_size, device=self.device)
        
        self.td_state['agents'] =  TensorDict(
                                    source={'capacity': self.td_state['capacity'],
                                            'speed': self.td_state['speed'],
                                            'cur_load': self.td_state['capacity'].clone(),
                                            'cur_time': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cur_node_idx': self.td_state['depot_idx'] * torch.ones((*batch_size, self.num_agents), dtype = torch.int64, device=self.device),
                                            'cur_ttime': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cum_ttime': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cur_length': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cum_length': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'trip_count': torch.zeros((*batch_size, self.num_agents), dtype = torch.int64, device=self.device),
                                            'visited_nodes': torch.zeros((*batch_size, self.num_agents, self.num_nodes), dtype=torch.bool, device=self.device),
                                            'action_mask': torch.ones((*batch_size, self.num_agents, self.num_nodes), dtype=torch.bool, device=self.device),
                                            # track whether the agent's previous action was visiting a depot
                                            'last_was_depot': torch.zeros((*batch_size, self.num_agents), dtype=torch.bool, device=self.device),
                                            'active_agents_mask': torch.ones((*batch_size, self.num_agents), dtype=torch.bool, device=self.device),
                                            'cur_step': torch.zeros((*batch_size, self.num_agents), dtype=torch.int32, device=self.device)},
                                    batch_size=batch_size, device=self.device)

        self.td_state['solution'] = TensorDict({}, batch_size=self.td_state.batch_size, device=self.device)

        if self.agent_selector is not None:
            self.agent_selector.set_env(self)
        self.obs_builder.set_env(self)
        self.reward_evaluator.set_env(self)

        done = self.td_state['done'].clone()
        reward = torch.zeros_like(done, dtype = torch.float, device=self.device)
        penalty = torch.zeros_like(done, dtype = torch.float, device=self.device)

        self.env_nsteps = 0
        return TensorDict(
            {
                "reward": reward,
                "penalty":penalty,
                "done": done,
            }, batch_size=batch_size, device=self.device)

    def reset_agent_select(self,
              num_agents:int|None=None,
              num_nodes:int|None=None,
              min_nodes: float = None,
              max_nodes: float = None,
              min_capacity: float = None,
              max_capacity: float = None,
              min_demand: int = None,
              max_demand: int = None,
              min_speed: float = None,
              max_speed: float = None,
              instance_name:str|None=None,
              sample_type:str='random',
              instance_dict:Dict=None,
              force_visit: bool = False,
              batch_size: Optional[torch.Size] = None,
              n_augment: Optional[int] = None,
              seed:int|None=None,
              device: Optional[str] = "cpu")-> TensorDict:
        """
        Resets the environment and sets the current agent.

        Args:
            num_agents(int, optional): Total number of agents. Defaults to None.
            num_nodes(int, optional): Total number of nodes. Defaults to None.
            capacity(float, optional): Total capacity for each agent. Defaults to None.
            speed(float, optional): Vehicles' speed. Defaults to None.
            instance_name(str, optional): Instance name. Defaults to None.
            sample_type(str): Sample type. It can be "random", "augment" or "saved". Defaults to "random".
            force_visit(bool): It forces the agent to visit all feasible nodes before going back to depot. Defaults to True.
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Data augmentation. Defaults to None.
            seed(int, optional): Random number generator seed. Defaults to None. 

        Returns:
            TensorDict: Environment information dictionary.
        """
        assert self.agent_selector is not None, f"this method requires an agent selector"

        td = self.reset(num_agents=num_agents, 
                            num_nodes=num_nodes, 
                            min_nodes=min_nodes,
                            max_nodes=max_nodes,
                            min_capacity=min_capacity,
                            max_capacity=max_capacity,
                            min_demand=min_demand,
                            max_demand=max_demand,
                            min_speed=min_speed,
                            max_speed=max_speed,
                            instance_name=instance_name, 
                            sample_type=sample_type,
                            instance_dict=instance_dict,
                            force_visit=force_visit,
                            batch_size=batch_size,
                            n_augment=n_augment,
                            seed=seed,
                            device=device)

        cur_agent_idx =  self.agent_selector._next_agent()
        td = self.set_cur_agent(cur_agent_idx, td)
        return td

    def reset_observe(self,
              num_agents:int|None=None,
              num_nodes:int|None=None,
              min_capacity:float|None=None,
              max_capacity:float|None=None,
              min_demand:int|None=None,
              max_demand:int|None=None,
              min_speed:float|None=None,
              max_speed:float|None=None,
              instance_name:str|None=None, 
              sample_type:str='random',
              instance_dict:Dict=None,
              force_visit: bool = False,
              batch_size: Optional[torch.Size] = None,
              n_augment: Optional[int] = None,
              seed:int|None=None,
              device: Optional[str] = "cpu",
              obs_list: Optional[List[str]] = ['agents_action_mask']) -> TensorDict:
        """
        Resets and observe the environment.

        Args:
            num_agents(int, optional): Total number of agents. Defaults to None.
            num_nodes(int, optional): Total number of nodes. Defaults to None.
            capacity(float, optional): Total capacity for each agent. Defaults to None.
            speed(float, optional): Vehicles' speed. Defaults to None.
            instance_name(str, optional): Instance name. Defaults to None.
            sample_type(str): Sample type. It can be "random", "augment" or "saved". Defaults to "random".
            force_visit(bool): It forces the agent to visit all feasible nodes before going back to depot. Defaults to True.
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Data augmentation. Defaults to None.
            seed(int, optional): Random number generator seed. Defaults to None.
            obs_list(List[str], optional): List of observations to be retrieved. Defaults to ['agents_action_mask'].

        Returns:
            TensorDict: Environment information dictionary.
        """

        td = self.reset(num_agents=num_agents, 
                            num_nodes=num_nodes, 
                            min_capacity=min_capacity,
                            max_capacity=max_capacity,
                            min_demand=min_demand,
                            max_demand=max_demand,
                            min_speed=min_speed,
                            max_speed=max_speed,
                            instance_name=instance_name, 
                            sample_type=sample_type,
                            instance_dict=instance_dict,
                            force_visit=force_visit,
                            batch_size=batch_size,
                            n_augment=n_augment,
                            seed=seed,
                            device=device)

        td = self.observe(td, obs_list)
        return td
    

    def reset_agent_select_observe(self,
              num_agents:int|None=None,
              num_nodes:int|None=None,
              min_capacity:float|None=None,
              max_capacity:float|None=None,
              min_demand:int|None=None,
              max_demand:int|None=None,
              min_speed:float|None=None,
              max_speed:float|None=None,
              instance_name:str|None=None, 
              sample_type:str='random',
              instance_dict:Dict=None,
              force_visit: bool = False,
              batch_size: Optional[torch.Size] = None,
              n_augment: Optional[int] = None,
              seed:int|None=None,
              device: Optional[str] = "cpu",
              obs_list: Optional[List[str]] = ["agent_cur_node_idx",'nodes_static', 'action_mask', 'agent']) -> TensorDict:
        """
        Resets the environment, sets the current agent and makes observations.

        Args:
            num_agents(int, optional): Total number of agents. Defaults to None.
            num_nodes(int, optional): Total number of nodes. Defaults to None.
            capacity(float, optional): Total capacity for each agent. Defaults to None.
            speed(float, optional): Vehicles' speed. Defaults to None.
            instance_name(str, optional): Instance name. Defaults to None.
            sample_type(str): Sample type. It can be "random", "augment" or "saved". Defaults to "random".
            force_visit(bool): It forces the agent to visit all feasible nodes before going back to depot. Defaults to True.
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Data augmentation. Defaults to None.
            seed(int, optional): Random number generator seed. Defaults to None. 

        Returns:
            TensorDict: Environment information dictionary.
        """
        assert self.agent_selector is not None, f"this method requires an agent selector"

        td = self.reset_agent_select(num_agents=num_agents, 
                            num_nodes=num_nodes, 
                            min_capacity=min_capacity,
                            max_capacity=max_capacity,
                            min_demand=min_demand,
                            max_demand=max_demand,
                            min_speed=min_speed,
                            max_speed=max_speed,
                            instance_name=instance_name, 
                            sample_type=sample_type,
                            instance_dict=instance_dict,
                            force_visit=force_visit,
                            batch_size=batch_size,
                            n_augment=n_augment,
                            seed=seed,
                            device=device)

        td = self.observe(td, obs_list)
        return td


    def _update_curr_agent_feasibility(self):

        """
        Update actions feasibility for current agent.
        
        Args:
            n/a.

        Returns:
            None.
        """
        eps = 1e-6
        _mask = self.td_state['nodes']['active_nodes_mask'].clone() 

        # capacity constraints
        c1 = self.td_state['demands'] <= self.td_state['cur_agent']['cur_load'] + eps

        _mask = _mask * c1 

        # after done close all services and open depot
        _mask = _mask * ~self.td_state['done'].unsqueeze(-1)
        _mask.scatter_(1, self.td_state['depot_idx'], True)

        if self.force_visit:
            can_visit = ~((self.td_state['cur_agent']['cur_node_idx'] == 0).squeeze(-1) & (_mask[:, 1:].sum(-1) > 0))
            _mask.scatter_(1, self.td_state['depot_idx'], can_visit.unsqueeze(-1))

        # update state
        self.td_state['cur_agent'].update({'action_mask': _mask}) 
        self.td_state['agents']['action_mask'].scatter_(1, 
                                            self.td_state['cur_agent_idx'][:,:,None].expand(-1,-1,self.num_nodes), _mask.unsqueeze(1))


    def _update_all_agents_feasibility(self):
        """
        Update actions feasibility for all agents simultaneously.
        
        Args:
            n/a.

        Returns:
            None.
        """

        # Get dimensions
        batch_size = self.td_state.batch_size  # This should be a torch.Size object
        
        # Get base mask from active nodes
        _mask = self.td_state['nodes']['active_nodes_mask'].clone()  # [B, N]
        
        # Add agent dimension
        _mask = _mask.unsqueeze(1)  # [B, 1, N]
        _mask = _mask.expand(*batch_size, self.num_agents, self.num_nodes)  # [B, num_agents, N]
        
        # Capacity constraints
        # demands: [B, N] -> [B, 1, N] -> [B, num_agents, N]
        demands = self.td_state['demands'].unsqueeze(1).expand(*batch_size, self.num_agents, self.num_nodes)
        # cur_load: [B, num_agents] -> [B, num_agents, 1]
        cur_load = self.td_state['agents']['cur_load'].unsqueeze(-1)
        
        c1 = demands <= cur_load
        _mask = _mask & c1

        # Zero out inactive agents (all their actions become unavailable)
        # Expand active_agents to cover all actions per agent
        active_expanded = self.td_state['agents']['active_agents_mask'].unsqueeze(-1).expand(-1, -1, self.num_nodes)
        _mask = _mask & active_expanded

        # After done, close all services but keep depot open for first agent
        done = self.td_state['done']  # [B, 1]
        _mask = _mask & ~done.unsqueeze(-1).unsqueeze(-1)  # [B, num_agents, N]

        # Open depot for agent 0 only in batch rows where `done` is True
        done_rows = done.squeeze(-1).nonzero(as_tuple=True)[0]  # indices of rows where all agents inactive
        if done_rows.numel() > 0:
            # ensure depot_idx is 1D [B]
            depot_idx = self.td_state['depot_idx'].squeeze(-1) if self.td_state['depot_idx'].dim() > 1 else self.td_state['depot_idx']
            # open depot only for agent 0 in those rows
            _mask[done_rows, 0, depot_idx[done_rows]] = True

        if self.force_visit:
            cur_node = self.td_state['agents']['cur_node_idx']  # [B, num_agents]
            depot_idx = self.td_state['depot_idx']  # [B, 1]

            at_depot = (cur_node == depot_idx)  # [B, num_agents]

            # Any unvisited service nodes (non-depot) across the batch
            has_unvisited = self.td_state['nodes']['active_nodes_mask'][:, 1:].any(dim=-1)  # [B]

            # Agents at depot with unvisited nodes must leave — close depot for them
            must_leave = at_depot & has_unvisited.unsqueeze(-1)  # [B, num_agents]

            # Close depot for agents that must leave
            depot_idx_expanded = depot_idx.unsqueeze(1).expand(*batch_size, self.num_agents, 1)  # [B, num_agents, 1]
            depot_open = _mask.gather(2, depot_idx_expanded) & ~must_leave.unsqueeze(-1)  # [B, num_agents, 1]
            _mask.scatter_(2, depot_idx_expanded, depot_open)

        self.td_state['agents']['action_mask'] = _mask
        
    def _update_done(self, action):

        """
        Update done state.

        Args:
            action(torch.Tensor): Tensor with agent moves.

        Returns:
            None.
        """

        former_done = self.td_state['done'].clone()

        # determine if current action visits a depot
        is_depot = action.eq(self.td_state['depot_idx'])
        # was the agent at depot in previous step?
        was_last = self.td_state['agents']['last_was_depot'].gather(1, self.td_state['cur_agent_idx']).clone()
        # agent is done only if it visits depot two consecutive times
        new_active = ~(was_last & is_depot)
        # update active mask for current agent
        self.td_state['agents']['active_agents_mask'].scatter_(1, self.td_state['cur_agent_idx'], new_active)
        # update last_was_depot flag for agent
        self.td_state['agents']['last_was_depot'].scatter_(1, self.td_state['cur_agent_idx'], is_depot)

        self.td_state['done'] = (~self.td_state['agents']['active_agents_mask']).all(dim=-1)
        self.td_state['done'][former_done] = True
        # update served nodes
        self.td_state['nodes']['active_nodes_mask'].scatter_(1, action, action.eq(self.td_state['depot_idx']))
        self.td_state['is_last_step'] = self.td_state['done'].eq(~former_done)


    def _update_state(self, action):

        """
        Update environment state.

        Args:
            action(torch.Tensor): Tensor with agent moves.

        Returns:
            None.
        """

        loc = self.td_state['coords'].gather(1, self.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        next_loc = self.td_state['coords'].gather(1, action[:,:,None].expand(-1, -1, 2))

        ptime = self.td_state['cur_agent']['cur_time'].clone()
        
        distance2j = torch.pairwise_distance(loc, next_loc, eps=0, keepdim = False)
        time2j = distance2j / self.td_state['cur_agent']['speed']
        if self.n_digits is not None:
            distance2j = torch.floor(self.n_digits * distance2j) / self.n_digits
            time2j = torch.floor(self.n_digits * time2j) / self.n_digits

        arrivej = ptime + time2j

        # update agent cur node
        self.td_state['cur_agent']['cur_node_idx'] = action
        self.td_state['agents']['cur_node_idx'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_node_idx'])
        # update agent cur time
        self.td_state['cur_agent']['cur_time'] = arrivej
        self.td_state['agents']['cur_time'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_time'])

        # update agent cum traveled time
        self.td_state['cur_agent']['cur_ttime'] = time2j
        self.td_state['cur_agent']['cum_ttime'] += time2j
        self.td_state['agents']['cur_ttime'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_ttime'])
        self.td_state['agents']['cum_ttime'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cum_ttime'])

        # update agent cum traveled length
        self.td_state['cur_agent']['cur_length'] = distance2j
        self.td_state['cur_agent']['cum_length'] += distance2j
        self.td_state['agents']['cur_length'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_length'])
        self.td_state['agents']['cum_length'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cum_length'])

        # When agent visits a depot, finalize trip and reset trip-local accumulators if trip allowed
        is_depot = action.eq(self.td_state['depot_idx'])
        if is_depot.any():
            # increment trip count for agents that visited depot
            self.td_state['agents']['trip_count'].scatter_add_(1, self.td_state['cur_agent_idx'], is_depot.float().to(torch.int64))
            # reset load at depot for multi-trip continuation
            self.td_state['cur_agent']['cur_load'] = torch.where(is_depot, self.td_state['agents']['capacity'].gather(1,self.td_state['cur_agent_idx']).squeeze(1), self.td_state['cur_agent']['cur_load'])
            self.td_state['agents']['cur_load'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_load'])


        # update agent load and node demands        
        self.td_state['cur_agent']['cur_load'] -= self.td_state['demands'].gather(1, action)

        # is agent is done set agent cur_load to 0
        agents_done = ~self.td_state['agents']['active_agents_mask'].gather(1, self.td_state['cur_agent_idx']).clone()
        self.td_state['cur_agent']['cur_load'] = torch.where( agents_done, 0., 
                                                             self.td_state['cur_agent']['cur_load'])
        
        self.td_state['nodes']['cur_demands'].scatter_(1, action, torch.zeros_like(action, dtype = torch.float))
        self.td_state['agents']['cur_load'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_load'])

        # update visited nodes
        r = torch.arange(*self.td_state.batch_size, device=self.device)
        self.td_state['agents']['visited_nodes'][r, self.td_state['cur_agent_idx'].squeeze(-1), action.squeeze(-1)] = True

        # update agent step
        self.td_state['cur_agent']['cur_step'] = torch.where(~agents_done, self.td_state['cur_agent']['cur_step']+1, 
                                                             self.td_state['cur_agent']['cur_step'])
        self.td_state['agents']['cur_step'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_step'])
        self.td_state['cur_node_idx'] = action.clone()

        # if all done activate first agent to guarantee batch consistency during agent sampling
        self.td_state['agents']['active_agents_mask'][self.td_state['agents']['active_agents_mask'].sum(1).eq(0), 0] = True

    def set_cur_agent(self, cur_agent_idx, td: TensorDict):
        """
        Set the current agent index.

        Args:
            agent_idx (int): The index of the agent to set as current.
        """
        agent_idx = cur_agent_idx
        assert self.td_state['agents']['active_agents_mask'].gather(1, agent_idx).all(), f"not feasible agent"

        self.td_state['cur_agent_idx'] = agent_idx
        self._update_cur_agent(agent_idx)
        agent_step = self.td_state['cur_agent']['cur_step']

        self.td_state['cur_agent_idx'] = agent_idx

        td["cur_agent_idx"] = self.td_state['cur_agent_idx'].clone()
        td["agent_step"] = agent_step    

        return td
    
    def _update_cur_agent(self, cur_agent_idx):

        """
        Update current agent.

        Args:
            cur_agent_idx(torch.Tensor): Current agent id.

        Returns:
            None.
        """
        self.td_state['cur_agent_idx'] =  cur_agent_idx
        self.td_state['cur_agent'] = TensorDict({
                        'action_mask': self.td_state['agents']['action_mask'].gather(1, self.td_state['cur_agent_idx'][:,:,None].expand(-1, -1, self.num_nodes)).squeeze(1),
                        'capacity': self.td_state['agents']['capacity'].gather(1, self.td_state['cur_agent_idx']).clone(),
                        'speed': self.td_state['agents']['speed'].gather(1, self.td_state['cur_agent_idx']).clone(),
                        'cur_load': self.td_state['agents']['cur_load'].gather(1, self.td_state['cur_agent_idx']).clone(),
                        'cur_time': self.td_state['agents']['cur_time'].gather(1, self.td_state['cur_agent_idx']).clone(),
                        'cur_node_idx': self.td_state['agents']['cur_node_idx'].gather(1, self.td_state['cur_agent_idx']).clone(),
                        'cur_ttime': self.td_state['agents']['cur_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                        'cum_ttime': self.td_state['agents']['cum_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                        'cur_length': self.td_state['agents']['cur_length'].gather(1, self.td_state['cur_agent_idx']).clone(),
                        'cum_length': self.td_state['agents']['cum_length'].gather(1, self.td_state['cur_agent_idx']).clone(),
                        'cur_step': self.td_state['agents']['cur_step'].gather(1, self.td_state['cur_agent_idx']).clone(),
                        }, batch_size=self.td_state.batch_size, device=self.device)
        
    def _update_solution(self, action):

        """
        Update agents and actions in solution.

        Args:
            action(torch.Tensor): Tensor with agent moves.

        Returns: 
            None.
        """

        # update solution dic
        if 'actions' in self.td_state['solution'].keys():
            self.td_state['solution','actions'] = torch.concat( [self.td_state['solution','actions'], action], dim=-1)
        else:
            self.td_state['solution','actions'] = action

        if 'agents' in self.td_state['solution'].keys():
            self.td_state['solution','agents'] = torch.concat( [self.td_state['solution','agents'], self.td_state['cur_agent_idx']], dim=-1)
        else:
            self.td_state['solution','agents'] = self.td_state['cur_agent_idx']

    def step(self, td: TensorDict) -> TensorDict:
        """
        Perform an environment step for active agent.

        Args:
            td(TensorDict): Environment tensor instance.

        Returns:
            td(TensorDict): Updated environment tensor instance.
        """

        if 'next_agent' in td.keys():
            agent_idx = td['next_agent']
            assert self.td_state['agents']['active_agents_mask'].gather(1, agent_idx).all(), f"not feasible agent"
            self._update_cur_agent(agent_idx)
            agent_step = self.td_state['cur_agent']['cur_step']
            td["agent_step"] = agent_step

        action = td["next_action"]
        assert self.td_state['cur_agent']['action_mask'].gather(1, action).all(), f"not feasible action"

        self._update_done(action)
        done = self.td_state['done'].clone()
        is_last_step = self.td_state['is_last_step'].clone()

        # update env state    
        self._update_state(action)

        # update solution dic
        self._update_solution(action)
        
        # get reward and penalty
        reward, penalty = self.reward_evaluator.get_reward(action)

        self.env_nsteps += 1
        td.update(
            {
                "reward": reward,
                "penalty":penalty,  
                "done": done,
                "is_last_step": is_last_step
            },
        )
        return td

    def step_observe(self, td: TensorDict,                              
                    obs_list: Optional[List[str]] = ['agents_action_mask']) -> TensorDict:

        """
        Perform an environment step for active agent.

        Args:
            td(TensorDict): Environment tensor instance.
            obs_list (Optional[List[str]]): List of observation keys to include. Defaults to ['agents_action_mask'].

        Returns:
            td(TensorDict): Updated environment tensor instance.
        """
        td = self.step(td)
        td = self.observe(td, obs_list=obs_list)
        return td

    def step_agent_select(self, td: TensorDict) -> TensorDict:
        """
        Perform an environment step for active agent.

        Args:
            td(TensorDict): Environment tensor instance.

        Returns:
            td(TensorDict): Updated environment tensor instance.
        """
        assert self.agent_selector is not None, f"this method requires an agent selector"

        td = self.step(td)

        # select and update cur agent
        cur_agent_idx =  self.agent_selector._next_agent()
        self._update_cur_agent(cur_agent_idx)
        agent_step = self.td_state['cur_agent']['cur_step']
        td["agent_step"] = agent_step
        return td

    def step_agent_select_observe(self, td: TensorDict,
                               obs_list: Optional[List[str]] = ['action_mask',  'agent', 'nodes_dynamic']) -> TensorDict:
        """
        Perform an environment step for active agent.

        Args:
            td(TensorDict): Environment tensor instance.

        Returns:
            td(TensorDict): Updated environment tensor instance.
        """
        assert self.agent_selector is not None, f"this method requires an agent selector"

        td = self.step_agent_select(td)
        td = self.observe(td, obs_list)
        return td

    def check_solution_validity(self):
        """
        Check if solution is valid according to HCVRP constraints, 
        allowing for depot reloads.
        """
        # 1. Initialize State
        batch_size = self.td_state['coords'].shape[0]
        curr_node = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        visited_nodes = torch.zeros(batch_size, self.num_nodes, dtype=torch.int64, device=self.device)
        
        # Track current load per batch item
        current_load = torch.zeros(batch_size, device=self.device)

        # 2. Sort actions by agent/time steps
        sorted_indices = torch.argsort(self.td_state['solution']['agents'], dim=-1, stable=True)
        sorted_actions = torch.gather(self.td_state['solution']['actions'], dim=-1, index=sorted_indices)
        
        # Get demands for all actions in sequence
        # (batch, seq_len)
        demands_seq = self.td_state['demands'].gather(1, sorted_actions)
        
        # Get the specific capacities for the agents assigned to these actions
        # (batch, seq_len)
        agent_ids = self.td_state['solution']['agents'].gather(1, sorted_indices)
        max_capacities = self.td_state['capacity'].gather(1, agent_ids)

        # 3. Validation Loop
        for ii in range(sorted_actions.size(1)):
            next_node = sorted_actions[:, ii]
            node_demand = demands_seq[:, ii]
            max_cap = max_capacities[:, ii]

            # --- CAPACITY & RELOAD LOGIC ---
            # If the agent is at the depot (node 0), their load resets to 0
            is_depot = (next_node == 0)
            current_load = torch.where(is_depot, torch.zeros_like(current_load), current_load)
            
            # Add demand of the next node
            current_load += node_demand

            # Check if capacity is exceeded
            assert torch.all(current_load <= max_cap), (
                f"Capacity exceeded at step {ii}. "
                f"Load: {current_load[current_load > max_cap]}, Max: {max_cap[current_load > max_cap]}"
            )

            # --- VISITATION LOGIC ---
            # Mark node as visited (ignore depot for multiple visits)
            fill = visited_nodes.gather(1, next_node.unsqueeze(-1))
            visited_nodes.scatter_(1, next_node.unsqueeze(-1), fill + 1)
            
            curr_node = next_node

        # 4. Final Constraints
        # Every customer node (1 to N) must be visited exactly once
        visited_nodes_exc_depot = visited_nodes[:, 1:]
        assert torch.all(visited_nodes_exc_depot == 1), "Some nodes were missed or visited multiple times!"
        #assert torch.all((visited_nodes_exc_depot == 0) | (visited_nodes_exc_depot == 1)), "Nodes were visited more than once!"