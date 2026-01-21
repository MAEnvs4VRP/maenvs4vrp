import torch
from tensordict import TensorDict

from typing import Tuple, Optional

import warnings

from maenvs4vrp.core.env_generator_builder import InstanceBuilder
from maenvs4vrp.core.env_observation_builder import ObservationBuilder
from maenvs4vrp.core.env_agent_selector import BaseSelector
from maenvs4vrp.core.env_agent_reward import RewardFn
from maenvs4vrp.core.env import AECEnv
from maenvs4vrp.utils.utils import gather_by_index


class Environment(AECEnv):
    """
    PCVRPTW environment generator class.
    """
    def __init__(self,
                instance_generator_object: InstanceBuilder,  
                obs_builder_object: ObservationBuilder,
                agent_selector_object: BaseSelector,
                reward_evaluator: RewardFn,
                seed=None,         
                device: Optional[str] = None,
                batch_size: torch.Size = None):
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
        self.env_name = 'pcvrptw'

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

        
    def observe(self, is_reset=False)-> TensorDict:
        """
        Retrieve agent environment observations.

        Args:
            is_reset(bool): If the environment is on reset. Defauts to False.

        Returns
            td_observations(TensorDict): Current agent observaions and masks dictionary.
        """

        self._update_feasibility()
        td_observations = self.obs_builder.get_observations(is_reset=is_reset)
        td_observations['action_mask'] = self.td_state['cur_agent']['action_mask'].clone()
        td_observations['agents_mask'] = self.td_state['agents']['active_agents_mask'].clone()
        return td_observations

    
    def sample_action(self, td: TensorDict)-> TensorDict:
        """
        Compute a random action from avaliable actions to current agent.
        
        Args:
            td(TensorDict): Environment instance tensor.

        Returns:
            td(TensorDict): Environment instance tensor with updated action.
        """
        action = torch.multinomial(self.td_state['cur_agent']["action_mask"].float(), 1).to(self.device)
        td['action'] = action
        return td


    def reset(self, 
              num_agents:int|None=None, 
              num_nodes:int|None=None, 
              capacity:int|None=None, 
              service_times:float|None=None, 
              profits:str='constant',              
              instance_name:str|None=None, 
              sample_type:str='random',
              batch_size: Optional[torch.Size] = None,
              n_augment: Optional[int] = None,
              seed:int|None=None,
              device: Optional[str] = "cpu")-> TensorDict:
        """
        Reset the environment.

        Args:
            num_agents(int, optional): Total number of agents. Defaults to None.
            num_nodes(int, optional): Total number of nodes. Defaults to None.
            capacity(int, optional): Total capacity for each agent. Defaults to None.
            service_times(float, optional): Service time in the nodes. Defaults to None.
            instance_name(str, optional): Instance name. Defaults to None.
            sample_type(str): Sample type. It can be "random", "augment" or "saved". Defaults to "random".
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Data augmentation. Defaults to None.
            seed(int, optional): Random number generator seed. Defaults to None. 

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

        instance_info = self.inst_generator.sample_instance(num_agents=num_agents, 
                                                            num_nodes=num_nodes, 
                                                            capacity=capacity, 
                                                            service_times=service_times, 
                                                            profits=profits,
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

        self.td_state['max_tour_duration'] =  self.td_state['end_time'] - self.td_state['start_time']

        time2depot = torch.pairwise_distance(self.td_state['depot_loc'], 
                                                              self.td_state['coords'], eps=0, keepdim = False)
        if self.n_digits is not None:
            time2depot = torch.floor(self.n_digits * time2depot) / self.n_digits

        self.td_state['time2depot'] = time2depot

        self.td_state['nodes'] = TensorDict(
                                    source={'cur_profits': self.td_state['profits'].clone(),
                                            'cur_demands': self.td_state['demands'].clone(),
                                            'active_nodes_mask': torch.ones((*batch_size, self.num_nodes),dtype=torch.bool, device=self.device)},
                                    batch_size=batch_size, device=self.device)
        self.td_state['agents'] =  TensorDict(
                                    source={'cum_profit': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'step_profit': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'capacity': self.td_state['capacity'],
                                            'cur_load': self.td_state['capacity'].clone() * torch.ones((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cur_time': self.td_state['start_time'].unsqueeze(1).clone() * torch.ones((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cur_node': self.td_state['depot_idx'] * torch.ones((*batch_size, self.num_agents), dtype = torch.int64, device=self.device),
                                            'cur_ttime': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cum_ttime': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'visited_nodes': torch.zeros((*batch_size, self.num_agents, self.num_nodes), dtype=torch.bool, device=self.device),
                                            'feasible_nodes': torch.ones((*batch_size, self.num_agents, self.num_nodes), dtype=torch.bool, device=self.device),
                                            'active_agents_mask': torch.ones((*batch_size, self.num_agents), dtype=torch.bool, device=self.device),
                                            'cur_step': torch.zeros((*batch_size, self.num_agents), dtype=torch.int32, device=self.device)},
                                    batch_size=batch_size, device=self.device)

        self.td_state['cur_agent_idx'] = torch.zeros((*batch_size, 1), dtype = torch.int64, device=self.device)
        self.td_state['cur_node_idx'] = self.td_state['depot_idx'].clone()

        self.td_state['cur_agent'] = TensorDict({
                                'action_mask': self.td_state['agents']['feasible_nodes'].gather(1, self.td_state['cur_agent_idx'][:,:,None].expand(-1, -1, self.num_nodes)).squeeze(1),
                                'cum_profit': self.td_state['agents']['cum_profit'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'step_profit': self.td_state['agents']['step_profit'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_load': self.td_state['agents']['cur_load'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_time': self.td_state['agents']['cur_time'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_node': self.td_state['agents']['cur_node'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_ttime': self.td_state['agents']['cur_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cum_ttime': self.td_state['agents']['cum_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_step': self.td_state['agents']['cur_step'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                }, batch_size=batch_size)
        
        self.td_state['solution'] = TensorDict({}, batch_size=batch_size)

        self.agent_selector.set_env(self)
        self.obs_builder.set_env(self)
        self.reward_evaluator.set_env(self)

        agent_step = self.td_state['cur_agent']['cur_step']
        done = self.td_state['done'].clone()
        reward = torch.zeros_like(done, dtype = torch.float, device=self.device)
        penalty = torch.zeros_like(done, dtype = torch.float, device=self.device)

        td_observations = self.observe(is_reset=True)
        self.env_nsteps = 0
        return TensorDict(
            {
                "agent_step": agent_step,
                "observations": td_observations,
                "cur_agent_idx":self.td_state['cur_agent_idx'].clone(),
                "cur_node_idx": self.td_state['cur_node_idx'].clone(),
                "reward": reward,
                "penalty":penalty,
                "done": done,
            },
            batch_size=batch_size, device=self.device)


    def _update_feasibility(self):
        """
        Update actions feasibility.
        
        Args:
            n/a.

        Returns:
            None.
        """

        _mask = self.td_state['nodes']['active_nodes_mask'].clone() 
        # time windows constraints
        loc = self.td_state['coords'].gather(1, self.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        ptime = self.td_state['cur_agent']['cur_time'].clone()
        time2j = torch.pairwise_distance(loc, self.td_state["coords"], eps=0, keepdim = False)
        if self.n_digits is not None:
            time2j = torch.floor(self.n_digits * time2j) / self.n_digits
        arrivej = ptime + time2j
        waitj = torch.clip(self.td_state['tw_low']-arrivej, min=0)
        service_startj = arrivej + waitj

        c1 = service_startj <= self.td_state['tw_high']
        c2 = service_startj + self.td_state['service_time'] + self.td_state['time2depot'] <= self.td_state['end_time'].unsqueeze(-1)

        # capacity constraints
        c3 = self.td_state['demands'] <= self.td_state['cur_agent']['cur_load']

        _mask = _mask * c1 * c2 * c3
        # update state
        self.td_state['cur_agent'].update({'action_mask': _mask}) 
        self.td_state['agents']['feasible_nodes'].scatter_(1, 
                                            self.td_state['cur_agent_idx'][:,:,None].expand(-1,-1,self.num_nodes), _mask.unsqueeze(1))


    def _update_done(self, action):
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
        loc = self.td_state['coords'].gather(1, self.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        next_loc = self.td_state['coords'].gather(1, action[:,:,None].expand(-1, -1, 2))

        ptime = self.td_state['cur_agent']['cur_time'].clone()
        time2j = torch.pairwise_distance(loc, next_loc, eps=0, keepdim = False)
        if self.n_digits is not None:
            time2j = torch.floor(self.n_digits * time2j) / self.n_digits
        tw = self.td_state['tw_low'].gather(1, action)
        service_time = self.td_state['service_time'].gather(1, action)

        arrivej = ptime + time2j
        waitj = torch.clip(tw-arrivej, min=0)

        time_update = arrivej + waitj + service_time
        # update agent cur node
        self.td_state['cur_agent']['cur_node'] = action
        self.td_state['agents']['cur_node'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_node'])
        # update agent cur time
        self.td_state['cur_agent']['cur_time'] = time_update

        # is agent is done set agent time to end_time
        agents_done = ~self.td_state['agents']['active_agents_mask'].gather(1, self.td_state['cur_agent_idx']).clone()
        self.td_state['cur_agent']['cur_time'] = torch.where(agents_done, self.td_state['end_time'].unsqueeze(-1), 
                                                             self.td_state['cur_agent']['cur_time'])
        self.td_state['agents']['cur_time'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_time'])

        # update agent cum traveled time
        self.td_state['cur_agent']['cur_ttime'] = time2j
        self.td_state['cur_agent']['cum_ttime'] += time2j
        self.td_state['agents']['cur_ttime'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_ttime'])
        self.td_state['agents']['cum_ttime'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cum_ttime'])

        # update agent load and node demands
        self.td_state['cur_agent']['cur_load'] -= self.td_state['demands'].gather(1, action)
        # is agent is done set agent cur_load to 0
        self.td_state['cur_agent']['cur_load'] = torch.where( agents_done, 0., 
                                                             self.td_state['cur_agent']['cur_load'])
        
        self.td_state['nodes']['cur_demands'].scatter_(1, action, torch.zeros_like(action, dtype = torch.float))
        self.td_state['agents']['cur_load'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_load'])
        # update visited nodes
        r = torch.arange(*self.td_state.batch_size, device=self.device)
        self.td_state['agents']['visited_nodes'][r, self.td_state['cur_agent_idx'].squeeze(-1), action.squeeze(-1)] = True

        # update agent profit
        self.td_state['cur_agent']['cum_profit'] += self.td_state['profits'].gather(1, action)
        self.td_state['agents']['cum_profit'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cum_profit'])
        self.td_state['cur_agent']['step_profit'] = self.td_state['profits'].gather(1, action)
        self.td_state['agents']['step_profit'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['step_profit'])
        self.td_state['nodes']['cur_profits'].scatter_(1, action, torch.zeros_like(action, dtype = torch.float))

        # update agent step
        self.td_state['cur_agent']['cur_step'] = torch.where(~agents_done, self.td_state['cur_agent']['cur_step']+1, 
                                                             self.td_state['cur_agent']['cur_step'])
        self.td_state['agents']['cur_step'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_step'])
        self.td_state['cur_node_idx'] = action.clone()
        # if all done activate first agent to guarantee batch consistency during agent sampling
        self.td_state['agents']['active_agents_mask'][self.td_state['agents']['active_agents_mask'].sum(1).eq(0), 0] = True

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
                                'action_mask': self.td_state['agents']['feasible_nodes'].gather(1, self.td_state['cur_agent_idx'][:,:,None].expand(-1, -1, self.num_nodes)).squeeze(1).clone(),
                                'cum_profit': self.td_state['agents']['cum_profit'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'step_profit': self.td_state['agents']['step_profit'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_load': self.td_state['agents']['cur_load'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_time': self.td_state['agents']['cur_time'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_node': self.td_state['agents']['cur_node'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_ttime': self.td_state['agents']['cur_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cum_ttime': self.td_state['agents']['cum_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
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
        action = td["action"]
        assert self.td_state['cur_agent']['action_mask'].gather(1, action).all(), f"not feasible action"

        self._update_done(action)
        done = self.td_state['done'].clone()
        is_last_step = self.td_state['is_last_step'].clone()

        # update env state    
        self._update_state(action)

        # update solution dic
        self. _update_solution(action)
        
        # get reward and penalty
        reward, penalty = self.reward_evaluator.get_reward(action)

        # select and update cur agent
        cur_agent_idx =  self.agent_selector._next_agent()
        self._update_cur_agent(cur_agent_idx)
        agent_step = self.td_state['cur_agent']['cur_step']

        # new observations
        td_observations = self.observe()
        self.env_nsteps += 1
        td.update(
            {
                "agent_step": agent_step,
                "observations": td_observations,
                "reward": reward,
                "penalty":penalty,  
                "cur_agent_idx":cur_agent_idx, 
                "cur_node_idx": self.td_state['cur_node_idx'].clone(),             
                "done": done,
                "is_last_step": is_last_step
            },
        )
        return td

    def check_solution_validity(self):
        """
        Check if solution is valid according to constraints.

        Args:
            N/a.

        Returns:
            None.
        """

        distance2depot = torch.pairwise_distance(self.td_state['coords'], self.td_state['coords'][..., 0:1, :], eps=0, keepdim=False)
        a = self.td_state['tw_low'] + distance2depot + self.td_state['service_time']  # Time to serve node and get back to depot
        b = self.td_state['tw_high'][:, 0, None]  # Depot late tw

        # Can agent serve node and get back to depot?
        assert torch.all(a <= b), "Agent cannot serve node and get back to depot."

        curr_node = torch.zeros(*self.batch_size, dtype=torch.int64, device=self.device)
        curr_time = torch.zeros(*self.batch_size, dtype=torch.float32, device=self.device)
        visited_nodes = torch.zeros(*self.batch_size, self.num_nodes, dtype=torch.int64, device=self.device)

        # Sort indices along each row
        sorted_indices = torch.argsort(self.td_state['solution']['agents'], dim=-1, stable=True)
        # Use gather to reorder data per row
        sorted_data = torch.gather(self.td_state['solution']['actions'], dim=-1, index=sorted_indices)

        demand = self.td_state['demands'].gather(1, sorted_data)

        for ii in range(sorted_data.size(1)):
            next_node = sorted_data[:, ii]
            curr_loc = gather_by_index(self.td_state['coords'], curr_node)
            next_loc = gather_by_index(self.td_state['coords'], next_node)
            dist = torch.pairwise_distance(curr_loc, next_loc, eps=0, keepdim=False)

            fill = visited_nodes.gather(1, next_node.unsqueeze(-1))
            visited_nodes.scatter_(1, next_node.unsqueeze(-1), fill + 1)

            curr_time = torch.max(curr_time + dist, gather_by_index(self.td_state['tw_low'], next_node))
            assert torch.all(curr_time <= gather_by_index(self.td_state['tw_high'], next_node)), "Agent must perform service before node's time window closes."

            curr_time = curr_time + gather_by_index(self.td_state['service_time'], next_node)
            curr_node = next_node
            curr_time[next_node == 0] = 0.0

        visited_nodes_exc_depot = visited_nodes[:, 1:]
        assert torch.all((visited_nodes_exc_depot == 0) | (visited_nodes_exc_depot == 1)), "Nodes were visited more than once!"
        
        # c3: capacity constraint
        used_cap = torch.zeros_like(self.td_state['demands'][:, 0])
        for ii in range(sorted_data.size(1)):
            #reset at depot
            used_cap = used_cap * (sorted_data[:, ii] != 0)
            used_cap += demand[:, ii]

            #Loads must no exceed capacity
            assert torch.all(used_cap <= self.td_state['capacity']), "Agent exceeded vehicle capacity."