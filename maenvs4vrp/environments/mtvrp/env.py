import torch
from tensordict import TensorDict

from typing import Optional, Dict, List

from maenvs4vrp.core.env_generator_builder import InstanceBuilder
from maenvs4vrp.core.env_observation_builder import ObservationBuilder
from maenvs4vrp.core.env_agent_selector import BaseSelector
from maenvs4vrp.core.env_agent_reward import RewardFn
from maenvs4vrp.core.env import AECEnv

from maenvs4vrp.utils.ops import gather_by_index, get_distance

MAX_TIME = 1_000_000
class Environment(AECEnv):
    """
    MTVRP environment generator class.
    """

    def __init__(
        self,
        instance_generator_object: InstanceBuilder,
        obs_builder_object: ObservationBuilder,
        agent_selector_object: BaseSelector,
        reward_evaluator: RewardFn,
        seed=None,
        device: Optional[str] = None,
        batch_size: torch.Size = None
    ):

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
        self.env_name = 'mtvrp'

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

        self.td_state = TensorDict({}, batch_size=self.batch_size, device=self.device) #Environment TensorDict

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
        if 'agent_cur_node_idx' in obs_list:
            td_observations['agent_cur_node_idx'] = self.td_state['cur_agent']['cur_node_idx'].clone()
        if 'agents_cur_node_idx' in obs_list:
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
    
    def reset(
        self,
        num_agents: int = None,
        num_nodes: int = None,
        min_coords: float = None,
        max_coords: float = None,
        capacity: int = None,
        service_time: float = None,
        instance_name:str|None=None,
        min_demands: int = None,
        max_demands: int = None,
        min_backhaul: int = None,
        max_backhaul: int = None,
        max_time: float = None,
        backhaul_ratio: float = None,
        backhaul_class: int = None,
        sample_backhaul_class: bool = None,
        max_distance_limit: float = None,
        speed: float = None,
        subsample: bool = True,
        variant_preset: str = None,
        use_combinations: bool = False,
        instance_dict:Dict=None,
        force_visit: bool = False,
        batch_size: Optional[torch.Size] = None,
        n_augment: Optional[int] = 2,
        sample_type: str = 'random',
        seed: int = None,
        device: Optional[str] = "cpu"
    ):
        
        """
        Reset the environment.

        Args:
            num_agents(int): Total number of agents. Defaults to None.
            num_nodes(int): Total number of nodes. Defaults to None.
            min_coords(float): Minimum number of coords. Defaults to None.
            max_coords(float): Maximum number of coords. Defaults to None.
            capacity(int): Vehicles' capacity. Defaults to None.
            service_time(float): Service time. Defaults to None.
            min_demands(int): Minimum number of demands. Defaults to None.
            max_demands(int): Maximum number of demands. Defaults to None.
            min_backhaul(int): Minimum number of backhauls. Defaults to None.
            max_backhaul(int): Maximum number of backhauls. Defaults to None.
            max_time(float): Maximum route time. Defaults to None.
            backhaul_ratio(float): Ratio of backhaul demands. Defaults to None.
            backhaul_class(int): Class of backhaul problem. If 1, it's unmixed, if 2, it's mixed. Defaults to None.
            sample_backhaul_class(bool): If backhaul class is sampled across batches. Defaults to False.
            max_distance_limit(float): Route distance limits. Defaults to None.
            speed(float): Vehicles' speed. Defaults to None.
            subsample(bool): If problem variants are to be sampled. Defaults to True.
            variant_preset(str): Variant preset to be sampled. Defaults to None.
            use_combinations(bool): It considers combinations for which sampling mask the instance is defined. Defaults to False.
            force_visit(bool): It forces the agent to visit all feasible nodes before going back to depot. Defaults to True.
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Number of augmentations. Defaults to None.
            sample_type(str): Type of instance to sample. It can be "random", "augment" or "saved". Defaults to "random".
            seed(int): Random number generator seed. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".

        Returns:
            TensorDict: Environment information dictionary.
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
            instance_info = self.inst_generator.sample_instance(
                num_agents = num_agents,
                num_nodes = num_nodes,
                min_coords = min_coords,
                max_coords = max_coords,
                capacity = capacity,
                service_time = service_time,
                instance_name=instance_name,
                min_demands = min_demands,
                max_demands = max_demands,
                min_backhaul = min_backhaul,
                max_backhaul = max_backhaul,
                max_time = max_time,
                backhaul_ratio = backhaul_ratio,
                backhaul_class = backhaul_class,
                sample_backhaul_class = sample_backhaul_class,
                max_distance_limit = max_distance_limit,
                speed = speed,
                subsample = subsample,
                variant_preset = variant_preset,
                use_combinations = use_combinations,
                sample_type=sample_type,
                n_augment=n_augment,
                batch_size = batch_size,
                seed = seed,
                device = device
            )

        self.num_nodes = instance_info['num_nodes']
        self.num_agents = instance_info['num_agents']

        if 'n_digits' in instance_info:
            self.n_digits = instance_info['n_digits'] 
        else:
            self.n_digits = None

        self.td_state = instance_info['data'].to(self.device) #Data from instance goes into env td_state

        self.td_state['done'] = torch.zeros(*batch_size, dtype=torch.bool)
        self.td_state['is_last_step'] = torch.zeros(*batch_size, dtype=torch.bool)
        self.td_state['depot_loc'] = self.td_state['coords'].gather(1, self.td_state['depot_idx'][:,:,None].expand(-1, -1, 2))

        self.td_state['start_time'] = self.td_state['tw_low'].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                          dtype=torch.int64, device=self.device)).squeeze(-1)
        self.td_state['end_time'] = self.td_state['tw_high'].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                        dtype=torch.int64, device=self.device)).squeeze(-1)


        self.td_state['max_tour_duration'] =  self.td_state['end_time'] - self.td_state['start_time']

        distance2depot = get_distance(self.td_state['depot_loc'], self.td_state['coords'])

        self.td_state['speed'] = instance_info['data']['speed'].clone()

        time2depot = distance2depot / self.td_state['speed']

        if self.n_digits is not None:
            distance2depot = torch.floor(self.n_digits * time2depot) / self.n_digits
            time2depot = torch.floor(self.n_digits * time2depot) / self.n_digits

        self.td_state['nodes'] = TensorDict(
                                    source={'linehaul_demands': self.td_state['linehaul_demands'].clone(),
                                            'backhaul_demands': self.td_state['backhaul_demands'].clone(),
                                            'distance2depot': distance2depot,
                                            'time2depot': time2depot,
                                            'active_nodes_mask': torch.ones((*batch_size, self.num_nodes),dtype=torch.bool, device=self.device)},
                                    batch_size=batch_size, device=self.device)
        
        self.td_state['agents'] =  TensorDict(
                                    source={'capacity': self.td_state['capacity'],
                                            'cur_time': self.td_state['start_time'].unsqueeze(1).clone() * torch.ones((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cur_node_idx': self.td_state['depot_idx'] * torch.ones((*batch_size, self.num_agents), dtype = torch.int64, device=self.device),
                                            'cur_ttime': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cum_ttime': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'visited_nodes': torch.zeros((*batch_size, self.num_agents, self.num_nodes), dtype=torch.bool, device=self.device),
                                            'action_mask': torch.ones((*batch_size, self.num_agents, self.num_nodes), dtype=torch.bool, device=self.device),
                                            'active_agents_mask': torch.ones((*batch_size, self.num_agents), dtype=torch.bool, device=self.device),
                                            'cur_step': torch.zeros((*batch_size, self.num_agents), dtype=torch.int32, device=self.device),
                                            'route_length': torch.zeros((*batch_size, self.num_agents), dtype=torch.float, device=self.device),
                                            'used_capacity_linehaul': torch.zeros((*batch_size, self.num_agents), dtype=torch.float32, device=self.device),
                                            'used_capacity_backhaul': torch.zeros((*batch_size, self.num_agents), dtype=torch.float32, device=self.device)},
                                    batch_size=batch_size, device=self.device)

        self.td_state['backhaul_class'] = instance_info['data']['backhaul_class'].clone()

        self.td_state['solution'] = TensorDict({}, batch_size=batch_size)

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
            },
            batch_size=batch_size, device=self.device)
    
    def reset_agent_select(self,
                        num_agents: int = None,
                        num_nodes: int = None,
                        min_coords: float = None,
                        max_coords: float = None,
                        capacity: int = None,
                        service_time: float = None,
                        instance_name:str|None=None,
                        min_demands: int = None,
                        max_demands: int = None,
                        min_backhaul: int = None,
                        max_backhaul: int = None,
                        max_time: float = None,
                        backhaul_ratio: float = None,
                        backhaul_class: int = None,
                        sample_backhaul_class: bool = None,
                        max_distance_limit: float = None,
                        speed: float = None,
                        subsample: bool = True,
                        variant_preset: str = None,
                        use_combinations: bool = False,
                        instance_dict:Dict=None,
                        force_visit: bool = False,
                        batch_size: Optional[torch.Size] = None,
                        n_augment: Optional[int] = None,
                        sample_type: str = 'random',
                        seed:int|None=None,
                        device: Optional[str] = "cpu")-> TensorDict:
        """
        Resets the environment and sets the current agent.

        Args:
            num_agents(int): Total number of agents. Defaults to None.
            num_nodes(int): Total number of nodes. Defaults to None.
            min_coords(float): Minimum number of coords. Defaults to None.
            max_coords(float): Maximum number of coords. Defaults to None.
            capacity(int): Vehicles' capacity. Defaults to None.
            service_time(float): Service time. Defaults to None.
            min_demands(int): Minimum number of demands. Defaults to None.
            max_demands(int): Maximum number of demands. Defaults to None.
            min_backhaul(int): Minimum number of backhauls. Defaults to None.
            max_backhaul(int): Maximum number of backhauls. Defaults to None.
            max_time(float): Maximum route time. Defaults to None.
            backhaul_ratio(float): Ratio of backhaul demands. Defaults to None.
            backhaul_class(int): Class of backhaul problem. If 1, it's unmixed, if 2, it's mixed. Defaults to None.
            sample_backhaul_class(bool): If backhaul class is sampled across batches. Defaults to False.
            max_distance_limit(float): Route distance limits. Defaults to None.
            speed(float): Vehicles' speed. Defaults to None.
            subsample(bool): If problem variants are to be sampled. Defaults to True.
            variant_preset(str): Variant preset to be sampled. Defaults to None.
            use_combinations(bool): It considers combinations for which sampling mask the instance is defined. Defaults to False.
            force_visit(bool): It forces the agent to visit all feasible nodes before going back to depot. Defaults to True.
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Number of augmentations. Defaults to None.
            sample_type(str): Type of instance to sample. It can be "random", "augment" or "saved". Defaults to "random".
            seed(int): Random number generator seed. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".

        Returns:
            TensorDict: Environment information dictionary.
        """
        assert self.agent_selector is not None, f"this method requires an agent selector"

        td = self.reset(num_agents = num_agents,
                        num_nodes = num_nodes,
                        min_coords = min_coords,
                        max_coords = max_coords,
                        capacity = capacity,
                        service_time = service_time,
                        instance_name=instance_name,
                        min_demands = min_demands,
                        max_demands = max_demands,
                        min_backhaul = min_backhaul,
                        max_backhaul = max_backhaul,
                        max_time = max_time,
                        backhaul_ratio = backhaul_ratio,
                        backhaul_class = backhaul_class,
                        sample_backhaul_class = sample_backhaul_class,
                        max_distance_limit = max_distance_limit,
                        speed = speed,
                        subsample = subsample,
                        variant_preset = variant_preset,
                        use_combinations = use_combinations,
                        sample_type=sample_type,
                        batch_size=batch_size,
                        n_augment=n_augment,
                        seed=seed,
                        device=device)

        cur_agent_idx =  self.agent_selector._next_agent()
        td = self.set_cur_agent(cur_agent_idx, td)
        return td

    def reset_observe(self,
                        num_agents: int = None,
                        num_nodes: int = None,
                        min_coords: float = None,
                        max_coords: float = None,
                        capacity: int = None,
                        service_time: float = None,
                        instance_name:str|None=None,
                        min_demands: int = None,
                        max_demands: int = None,
                        min_backhaul: int = None,
                        max_backhaul: int = None,
                        max_time: float = None,
                        backhaul_ratio: float = None,
                        backhaul_class: int = None,
                        sample_backhaul_class: bool = None,
                        max_distance_limit: float = None,
                        speed: float = None,
                        subsample: bool = True,
                        variant_preset: str = None,
                        use_combinations: bool = False,
                        instance_dict:Dict=None,
                        force_visit: bool = False,
                        batch_size: Optional[torch.Size] = None,
                        n_augment: Optional[int] = None,
                        sample_type: str = 'random',
                        seed:int|None=None,
                        device: Optional[str] = "cpu",
                        obs_list: Optional[List[str]] = ['agents_action_mask']) -> TensorDict:
        """
        Resets and observe the environment.

        Args:
            num_agents(int): Total number of agents. Defaults to None.
            num_nodes(int): Total number of nodes. Defaults to None.
            min_coords(float): Minimum number of coords. Defaults to None.
            max_coords(float): Maximum number of coords. Defaults to None.
            capacity(int): Vehicles' capacity. Defaults to None.
            service_time(float): Service time. Defaults to None.
            min_demands(int): Minimum number of demands. Defaults to None.
            max_demands(int): Maximum number of demands. Defaults to None.
            min_backhaul(int): Minimum number of backhauls. Defaults to None.
            max_backhaul(int): Maximum number of backhauls. Defaults to None.
            max_time(float): Maximum route time. Defaults to None.
            backhaul_ratio(float): Ratio of backhaul demands. Defaults to None.
            backhaul_class(int): Class of backhaul problem. If 1, it's unmixed, if 2, it's mixed. Defaults to None.
            sample_backhaul_class(bool): If backhaul class is sampled across batches. Defaults to False.
            max_distance_limit(float): Route distance limits. Defaults to None.
            speed(float): Vehicles' speed. Defaults to None.
            subsample(bool): If problem variants are to be sampled. Defaults to True.
            variant_preset(str): Variant preset to be sampled. Defaults to None.
            use_combinations(bool): It considers combinations for which sampling mask the instance is defined. Defaults to False.
            force_visit(bool): It forces the agent to visit all feasible nodes before going back to depot. Defaults to True.
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Number of augmentations. Defaults to None.
            sample_type(str): Type of instance to sample. It can be "random", "augment" or "saved". Defaults to "random".
            seed(int): Random number generator seed. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".
            obs_list(List[str], optional): List of observations to be retrieved. Defaults to ['agents_action_mask'].

        Returns:
            TensorDict: Environment information dictionary.
        """

        td = self.reset(num_agents = num_agents,
                        num_nodes = num_nodes,
                        min_coords = min_coords,
                        max_coords = max_coords,
                        capacity = capacity,
                        service_time = service_time,
                        instance_name=instance_name,
                        min_demands = min_demands,
                        max_demands = max_demands,
                        min_backhaul = min_backhaul,
                        max_backhaul = max_backhaul,
                        max_time = max_time,
                        backhaul_ratio = backhaul_ratio,
                        backhaul_class = backhaul_class,
                        sample_backhaul_class = sample_backhaul_class,
                        max_distance_limit = max_distance_limit,
                        speed = speed,
                        subsample = subsample,
                        variant_preset = variant_preset,
                        use_combinations = use_combinations,
                        sample_type=sample_type,
                        batch_size=batch_size,
                        n_augment=n_augment,
                        seed=seed,
                        device=device)

        td = self.observe(td, obs_list)
        return td
    

    def reset_agent_select_observe(self,
                        num_agents: int = None,
                        num_nodes: int = None,
                        min_coords: float = None,
                        max_coords: float = None,
                        capacity: int = None,
                        service_time: float = None,
                        instance_name:str|None=None,
                        min_demands: int = None,
                        max_demands: int = None,
                        min_backhaul: int = None,
                        max_backhaul: int = None,
                        max_time: float = None,
                        backhaul_ratio: float = None,
                        backhaul_class: int = None,
                        sample_backhaul_class: bool = None,
                        max_distance_limit: float = None,
                        speed: float = None,
                        subsample: bool = True,
                        variant_preset: str = None,
                        use_combinations: bool = False,
                        instance_dict:Dict=None,
                        force_visit: bool = False,
                        batch_size: Optional[torch.Size] = None,
                        n_augment: Optional[int] = None,
                        sample_type: str = 'random',
                        seed:int|None=None,
                        device: Optional[str] = "cpu",
                        obs_list: Optional[List[str]] = ["agent_cur_node_idx",'nodes_static', 'action_mask', 'agent']) -> TensorDict:
        """
        Resets the environment, sets the current agent and makes observations.

        Args:
            num_agents(int): Total number of agents. Defaults to None.
            num_nodes(int): Total number of nodes. Defaults to None.
            min_coords(float): Minimum number of coords. Defaults to None.
            max_coords(float): Maximum number of coords. Defaults to None.
            capacity(int): Vehicles' capacity. Defaults to None.
            service_time(float): Service time. Defaults to None.
            min_demands(int): Minimum number of demands. Defaults to None.
            max_demands(int): Maximum number of demands. Defaults to None.
            min_backhaul(int): Minimum number of backhauls. Defaults to None.
            max_backhaul(int): Maximum number of backhauls. Defaults to None.
            max_time(float): Maximum route time. Defaults to None.
            backhaul_ratio(float): Ratio of backhaul demands. Defaults to None.
            backhaul_class(int): Class of backhaul problem. If 1, it's unmixed, if 2, it's mixed. Defaults to None.
            sample_backhaul_class(bool): If backhaul class is sampled across batches. Defaults to False.
            max_distance_limit(float): Route distance limits. Defaults to None.
            speed(float): Vehicles' speed. Defaults to None.
            subsample(bool): If problem variants are to be sampled. Defaults to True.
            variant_preset(str): Variant preset to be sampled. Defaults to None.
            use_combinations(bool): It considers combinations for which sampling mask the instance is defined. Defaults to False.
            force_visit(bool): It forces the agent to visit all feasible nodes before going back to depot. Defaults to True.
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Number of augmentations. Defaults to None.
            sample_type(str): Type of instance to sample. It can be "random", "augment" or "saved". Defaults to "random".
            seed(int): Random number generator seed. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".
            seed(int, optional): Random number generator seed. Defaults to None.
            obs_list(list, optional): List of observations to include. Defaults to None.

        Returns:
            TensorDict: Environment information dictionary.
        """
        assert self.agent_selector is not None, f"this method requires an agent selector"

        td = self.reset_agent_select(num_agents = num_agents,
                        num_nodes = num_nodes,
                        min_coords = min_coords,
                        max_coords = max_coords,
                        capacity = capacity,
                        service_time = service_time,
                        instance_name=instance_name,
                        min_demands = min_demands,
                        max_demands = max_demands,
                        min_backhaul = min_backhaul,
                        max_backhaul = max_backhaul,
                        max_time = max_time,
                        backhaul_ratio = backhaul_ratio,
                        backhaul_class = backhaul_class,
                        sample_backhaul_class = sample_backhaul_class,
                        max_distance_limit = max_distance_limit,
                        speed = speed,
                        subsample = subsample,
                        variant_preset = variant_preset,
                        use_combinations = use_combinations,
                        sample_type=sample_type,
                        n_augment=n_augment,
                        batch_size=batch_size,
                        seed=seed,
                        device=device)

        td = self.observe(td, obs_list)
        return td

    def _update_curr_agent_feasibility(self):

        """
        Update actions feasibility.
        
        Args:
            n/a.

        Returns:
            None.
        """

        active_nodes = self.td_state['nodes']['active_nodes_mask'].clone() #Active nodes. Agent can only visit node if it's active
        loc = self.td_state['coords'].gather(1, self.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2)) #Current agent location
        ptime = self.td_state['cur_agent']['cur_time'].clone() #Agent current time

        #Distance between current agent and nodes
        distance2j = get_distance(loc, self.td_state["coords"])
        time2j = distance2j / self.td_state['speed']
        if self.n_digits is not None:
            distance2j = torch.floor(self.n_digits * distance2j) / self.n_digits
            time2j = torch.floor(self.n_digits * time2j) / self.n_digits


        time2depot = self.td_state['nodes']['time2depot'].clone() #Time from nodes to depot
        distance2depot = self.td_state['nodes']['distance2depot'].clone()
        time2arrive = distance2j / self.td_state['speed']
        arrival_time = ptime + time2arrive #Arrival time. Current time + time 2 arrive (distance / speed)

        #Constraint 1. Can arrive to node in time.
        c1 = arrival_time <= self.td_state['tw_high']
        #Constraint 2. If problem is closed, if agent can arrive to depot in time.
        c2 = (torch.max(arrival_time, self.td_state['tw_low']) + self.td_state['service_time'] + time2depot) * ~self.td_state['open_routes'] <= self.td_state['end_time'].unsqueeze(-1)
        #Constraint 3. Does agent exceed distance limit.
        c3 = self.td_state['cur_agent']['cur_route_length'] + distance2j + (distance2depot * ~self.td_state['open_routes']) <= self.td_state['distance_limits']

        #Demands constraints

        #Capacity

        exceeds_cap_linehaul = self.td_state['linehaul_demands'] + self.td_state['cur_agent']['used_capacity_linehaul'] > self.td_state['agents']['capacity']
        exceeds_cap_backhaul = self.td_state['backhaul_demands'] + self.td_state['cur_agent']['used_capacity_backhaul'] > self.td_state['agents']['capacity']

        '''
        Backhaul class 1. Node either linehaul or backhaul. Linehauls before backhauls.
        '''

        linehaul_missing = ((self.td_state['linehaul_demands'] * active_nodes).sum(-1) > 0).unsqueeze(-1)
        is_carrying_backhaul = gather_by_index(src=self.td_state['backhaul_demands'], idx=self.td_state['cur_agent']['cur_node_idx'], dim=1, squeeze=False) > 0
        meets_demand_constraint_backhaul_1 = (linehaul_missing & ~exceeds_cap_linehaul & ~is_carrying_backhaul & (self.td_state['linehaul_demands'] > 0)) | (~exceeds_cap_backhaul & (self.td_state['backhaul_demands'] > 0))

        '''
        Backhaul class 2. Mixed linehauls and backhauls
        '''

        cannot_serve_linehaul = self.td_state['linehaul_demands'] > self.td_state['capacity'] - self.td_state['cur_agent']['used_capacity_backhaul']
        meets_demand_constraint_backhaul_2 = ~exceeds_cap_linehaul & ~exceeds_cap_backhaul & ~cannot_serve_linehaul

        #Demand constraints according to backhaul class

        meet_demand_constraints = ((self.td_state['backhaul_class'] == 1) & meets_demand_constraint_backhaul_1) | ((self.td_state['backhaul_class'] == 2) & meets_demand_constraint_backhaul_2)
        _mask = active_nodes & c1 & c2 & c3 & meet_demand_constraints

        # after done close all services and open depot
        _mask = _mask * ~self.td_state['done'].unsqueeze(-1)
        _mask.scatter_(1, self.td_state['depot_idx'], True)

        if self.force_visit:
            can_visit = ~((self.td_state['cur_agent']['cur_node_idx'] == 0).squeeze(-1) & (_mask[:, 1:].sum(-1) > 0))
            _mask.scatter_(1, self.td_state['depot_idx'], can_visit.unsqueeze(-1))

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
        raise NotImplementedError()
    
    def _update_done(
        self,
        action
    ):
        
        """
        Update done state.

        Args:
            action(torch.Tensor): Tensor with agent moves.

        Returns:
            None.
        """
        
        former_done = self.td_state['done'].clone()

        # update done agents
        self.td_state['agents']['active_agents_mask'].scatter_(1, self.td_state['cur_agent_idx'], 
                                                                    ~action.eq(self.td_state['depot_idx']))
        
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

        distance2j = get_distance(loc, next_loc)
        time2j = distance2j / self.td_state['speed']
        if self.n_digits is not None:
            distance2j = torch.floor(self.n_digits * distance2j) / self.n_digits
            time2j = torch.floor(self.n_digits * time2j) / self.n_digits

        tw = self.td_state['tw_low'].gather(1, action)
        service_time = self.td_state['service_time'].gather(1, action)

        arrivej = ptime + time2j
        waitj = torch.clip(tw-arrivej, min=0)

        time_update = arrivej + waitj + service_time

        is_open_and_getting_to_depot = (self.td_state['open_routes']) & (action.eq(self.td_state['depot_idx']))

        #Update distances and time if problem is open and agent going back to depot
        distance2j[is_open_and_getting_to_depot] = 0.
        time2j[is_open_and_getting_to_depot] = 0.

        # update agent cur node
        self.td_state['cur_agent']['cur_node_idx'] = action
        self.td_state['agents']['cur_node_idx'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_node_idx'])
        
        # update agent cur time
        self.td_state['cur_agent']['cur_time'] = time_update
        self.td_state['agents']['cur_time'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_time'])

        #Current route length
        self.td_state['cur_agent']['cur_route_length'] += distance2j
        self.td_state['agents']['route_length'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_route_length'])    

        # update agent cum traveled time
        self.td_state['cur_agent']['cur_ttime'] = time2j
        self.td_state['cur_agent']['cum_ttime'] += time2j
        self.td_state['agents']['cur_ttime'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_ttime'])
        self.td_state['agents']['cum_ttime'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cum_ttime'])
        
        self.td_state['nodes']['linehaul_demands'].scatter_(1, action, torch.zeros_like(action, dtype = torch.float))
        self.td_state['nodes']['backhaul_demands'].scatter_(1, action, torch.zeros_like(action, dtype = torch.float))
        # update visited nodes
        r = torch.arange(*self.td_state.batch_size, device=self.device)
        self.td_state['agents']['visited_nodes'][r, self.td_state['cur_agent_idx'].squeeze(-1), action.squeeze(-1)] = True

        # update agent step
        agents_done = ~self.td_state['agents']['active_agents_mask'].gather(1, self.td_state['cur_agent_idx']).clone()
        self.td_state['cur_agent']['cur_step'] = torch.where(~agents_done, self.td_state['cur_agent']['cur_step']+1, 
                                                             self.td_state['cur_agent']['cur_step'])
        self.td_state['agents']['cur_step'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_step'])

        # update used capacities
        selected_demand_linehaul = gather_by_index(src=self.td_state['linehaul_demands'], idx=self.td_state['cur_agent']['cur_node_idx'], dim=1, squeeze=False)
        selected_demand_backhaul = gather_by_index(src=self.td_state['backhaul_demands'], idx=self.td_state['cur_agent']['cur_node_idx'], dim=1, squeeze=False)
        cur_node = self.td_state['agents']['cur_node_idx'].gather(1, self.td_state['cur_agent_idx']).clone()
        used_capacity_linehaul = (self.td_state['cur_agent']['used_capacity_linehaul'] + selected_demand_linehaul)
        used_capacity_backhaul = (self.td_state['cur_agent']['used_capacity_backhaul'] + selected_demand_backhaul)
        self.td_state['cur_agent']['used_capacity_linehaul'] = used_capacity_linehaul
        self.td_state['cur_agent']['used_capacity_backhaul'] = used_capacity_backhaul
        self.td_state['agents']['used_capacity_linehaul'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['used_capacity_linehaul'])
        self.td_state['agents']['used_capacity_backhaul'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['used_capacity_backhaul'])

        # if all done activate first agent to guarantee batch consistency during agent sampling
        self.td_state['agents']['active_agents_mask'][self.td_state['agents']['active_agents_mask'].sum(1).eq(0), 0] = True
        self.td_state['cur_node_idx'] = action.clone()
        self.td_state['agents']['active_agents_mask']

    def set_cur_agent(self, cur_agent_idx, td: TensorDict):
        """
        Set and update the next active agent.

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
                                'action_mask': self.td_state['agents']['action_mask'].gather(1, self.td_state['cur_agent_idx'][:,:,None].expand(-1, -1, self.num_nodes)).squeeze(1).clone(),
                                'cur_agent_idx': cur_agent_idx,
                                'cur_route_length': self.td_state['agents']['route_length'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_time': self.td_state['agents']['cur_time'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_node_idx': self.td_state['agents']['cur_node_idx'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_ttime': self.td_state['agents']['cur_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cum_ttime': self.td_state['agents']['cum_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_step': self.td_state['agents']['cur_step'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'used_capacity_linehaul': self.td_state['agents']['used_capacity_linehaul'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'used_capacity_backhaul': self.td_state['agents']['used_capacity_backhaul'].gather(1, self.td_state['cur_agent_idx']).clone()
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
        Check if solution is valid according to constraints.

        Args:
            N/a.

        Returns:
            None.
        """

        distance2depot = get_distance(self.td_state['coords'], self.td_state['coords'][..., 0:1, :])
        time2depot = distance2depot / self.td_state['speed']
        if self.n_digits is not None:
            distance2depot = torch.floor(self.n_digits * distance2depot) / self.n_digits
            time2depot = torch.floor(self.n_digits * time2depot) / self.n_digits

        a = self.td_state['tw_low'] + time2depot + self.td_state['service_time'] #Time 2 serve node and get back to depot
        b = self.td_state['time_windows'][..., 0, 1, None] #Depot late tw

        #Can agent serve node and get back to depot?
        assert torch.all(a <= b), "Agent cannot serve node and get back to depot."

        #Actions cycle assert. Curr_node starts at 0 (depot) and iteratively keeps going onto the next. 
        curr_node = torch.zeros(*self.batch_size, dtype=torch.int64, device=self.device)
        curr_time = torch.zeros(*self.batch_size, dtype=torch.float32, device=self.device)
        curr_length = torch.zeros(*self.batch_size, dtype=torch.float32, device=self.device)
        visited_nodes = torch.zeros(*self.batch_size, self.num_nodes, dtype=torch.int64, device=self.device)
        # Sort indices along each row
        sorted_indices = torch.argsort(self.td_state['solution']['agents'], dim=-1, stable=True)
        # Use gather to reorder data per row
        sorted_data = torch.gather(self.td_state['solution']['actions'], dim=-1, index=sorted_indices)
        
        for ii in range(sorted_data.size(1)):
            next_node = sorted_data[:, ii]
            curr_loc = gather_by_index(self.td_state['coords'], curr_node)
            next_loc = gather_by_index(self.td_state['coords'], next_node)
            dist = torch.pairwise_distance(curr_loc, next_loc, eps=0, keepdim = False)
            
            fill = visited_nodes.gather(1, next_node.unsqueeze(-1))
            visited_nodes.scatter_(1, next_node.unsqueeze(-1), fill + 1)

            curr_length = curr_length + dist * ~(self.td_state['open_routes'].squeeze(-1) & (next_node == 0)) #Update curr_length
            assert torch.all(curr_length <= self.td_state['distance_limits'].squeeze(-1)), "Route length exceeds distance limit."
            curr_length[next_node == 0] = 0.0 #Reset length for depot

            curr_time = torch.max(curr_time + dist, gather_by_index(self.td_state['time_windows'], next_node)[..., 0]) #Curr time either time to get to node or early tw
            assert torch.all(curr_time <= gather_by_index(self.td_state['time_windows'], next_node)[..., 1]), "Agent must perform service before node's time window closes."

            curr_time = curr_time + gather_by_index(self.td_state['service_time'], next_node)
            curr_node = next_node
            curr_time[next_node == 0] = 0.0

        visited_nodes_exc_depot = visited_nodes[:, 1:]
        assert(torch.all((visited_nodes_exc_depot == 0) | (visited_nodes_exc_depot == 1))), "Nodes were visited more than once!"

        demand_l = self.td_state['linehaul_demands'].gather(1, sorted_data)
        demand_b = self.td_state['backhaul_demands'].gather(1, sorted_data)

        used_cap_l = torch.zeros_like(self.td_state['linehaul_demands'][:, 0]) #Starts at 0
        used_cap_b = torch.zeros_like(self.td_state['backhaul_demands'][:, 0]) #Starts at 0

        for ii in range(sorted_data.size(1)):
            #reset at depot
            used_cap_l = used_cap_l * (sorted_data[:, ii] != 0)
            used_cap_b = used_cap_b * (sorted_data[:, ii] != 0)

            used_cap_l += demand_l[:, ii]
            used_cap_b += demand_b[:, ii]

            #Backhaul class 1 (unmixed), agents cannot supply linehaul if carrying backhaul
            assert(
                (self.td_state['backhaul_class'] == 2) |
                (used_cap_b == 0) |
                ((self.td_state['backhaul_class'] == 1) & ~(demand_l[:, ii] > 0))
            ).all(), "Cannot pickup linehaul while carrying backhaul in unmixed problems."

            #Backhaul class 2 (mixed), agents cannot supply linehaul, if backhaul load + linehaul demand in node exceeds agent's capacity

            assert(
                (self.td_state['backhaul_class'] == 1) |
                (used_cap_b == 0) |
                ((self.td_state['backhaul_class'] == 2) & (used_cap_b + demand_l[:, ii] <= self.td_state['capacity']))
            ).all(), "Cannot supply linehaul, not enough load."

            #Loads must not exceed capacity
            assert(
                used_cap_l <= self.td_state['capacity']
            ).all(), "Used more linehaul than capacity: {}/{}".format(used_cap_l, self.td_state['capacity'])

            assert(
                used_cap_b <= self.td_state['capacity']
            ).all(), "Used more backhaul than capacity: {}/{}".format(used_cap_b, self.td_state['capacity'])
