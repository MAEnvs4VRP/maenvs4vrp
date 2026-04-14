import torch
from tensordict import TensorDict

from maenvs4vrp.core.env_observation_builder import ObservationBuilder
from maenvs4vrp.core.env import AECEnv

from typing import Optional, Dict


class Observations(ObservationBuilder):
    """
    GMTDVRP observations class.
    """

    POSSIBLE_NODES_STATIC_FEATURES = ['x_coordinate', 'y_coordinate', 'tw_low',
                                    'tw_high', 'linehaul_demand', 'backhaul_demand', 'service_time', 'tw_high_minus_tw_low_div_max_dur',
                                    'x_coordinate_min_max', 'y_coordinate_min_max', 'is_depot']

    POSSIBLE_NODES_DYNAMIC_FEATURES = ['time2open_div_end_time', 'time2close_div_end_time', 'arrive2node_div_end_time',
                                'time2open_after_step_div_end_time', 'time2close_after_step_div_end_time',
                                'time2end_after_step_div_end_time', 'fract_time_after_step_div_end_time',
                                'reachable_frac_agents']

    POSSIBLE_AGENT_FEATURES = ['x_coordinate', 'y_coordinate','x_coordinate_min_max', 'y_coordinate_min_max', 'frac_current_time', 
                                'arrivedepot_div_end_time', 'cur_linehaul_load', 'cur_backhaul_load', 'available_load',
                                'available_load_vrpmpd' ,'remaining_dist', 'frac_feasible_nodes', 'current_time']

    POSSIBLE_OTHER_AGENTS_FEATURES = ['x_coordinate', 'y_coordinate','x_coordinate_min_max', 'y_coordinate_min_max', 'frac_current_time', 
                                    'dist2depot_div_end_time', 
                                    'dist2agent_div_end_time', 'frac_feasible_nodes','time_delta2agent_div_max_dur', 'was_last']

    POSSIBLE_GLOBAL_FEATURES = [ 'frac_linehaul_demands', 'frac_backhaul_demands',
                                'frac_done_agents', 'open_routes', 'distance_limits', 
                                'max_tw_depot', 'is_problem_backhaul_mixed']


    def __init__(self, feature_list:Dict = None):
        super().__init__()
        """
        Constructor.

        Args:
            feature_list(Dict): Dictionary containing observation features list to be available to the agent. Defaults to None.

        Returns:
            None.
        """

        self.default_feature_list = {'nodes_static': {'x_coordinate': {'feat': 'x_coordinate', 'norm': None},
                                    'y_coordinate': {'feat': 'y_coordinate', 'norm': None},
                                    'demand': {'feat': 'linehaul_demand', 'norm': None},
                                    'is_depot': {'feat': 'is_depot', 'norm': None}},
                                    'nodes_dynamic': [],
                                    'agent': ['current_time'],
                                    'other_agents': [],
                                    'all_agents': [],
                                    'global': [ 'frac_linehaul_demands']}

        if feature_list is None:
            feature_list = self.default_feature_list

        self.feature_list = feature_list
        self.possible_nodes_static_features = self.POSSIBLE_NODES_STATIC_FEATURES
        self.possible_nodes_dynamic_features = self.POSSIBLE_NODES_DYNAMIC_FEATURES
        self.possible_agent_features = self.POSSIBLE_AGENT_FEATURES
        self.possible_other_agents_features = self.POSSIBLE_OTHER_AGENTS_FEATURES
        self.possible_all_agents_features = self.POSSIBLE_ALL_AGENTS_FEATURES
        self.possible_global_features = self.POSSIBLE_GLOBAL_FEATURES

    def set_env(self, env:AECEnv):   

        """
        Set environment.

        Args:
            env(AECEnv): Environment.

        Returns:
            None.
        """

        super().set_env(env)

    
    def get_nodes_static_feat_dim(self):
        """
        Nodes static features dimensions.

        Args:
            n/a.

        Returns:
            int: Nodes static features dimensions.
        """
        return sum([self.feature_list.get('nodes_static', []).get(f).get('dim', 1) \
                    for f in self.feature_list.get('nodes_static')])

    def get_nodes_dynamic_feat_dim(self):
        """
        Nodes dynamic features dimensions.

        Args:
            n/a.

        Returns:
            int: Nodes dynamic features dimensions.
        """
        return len(self.feature_list.get('nodes_dynamic', []))

    def get_nodes_feat_dim(self):
        """
        Nodes features dimensions.

        Args:
            n/a.

        Returns:
            int: Nodes features dimensions.
        """
        return self.get_nodes_static_feat_dim()+self.get_nodes_dynamic_feat_dim()

    def get_agent_feat_dim(self):
        """
        Agent features dimensions.

        Args:
            n/a.

        Returns:
            int: Agent features dimensions.
        """
        return len(self.feature_list.get('agent', []))

    def get_other_agents_feat_dim(self):
        """
        Other agent features dimensions.

        Args:
            n/a.

        Returns:
            int: Other agent features dimensions.
        """
        return len(self.feature_list.get('other_agents', []))

    def get_global_feat_dim(self):
        """
        Global features dimensions.

        Args:
            n/a.

        Returns:
            int: Global features dimensions.
        """
        return len(self.feature_list.get('global', []))


    ## static features
    def get_feat_x_coordinate(self):
        """
        Instance nodes X coordinates.

        Args:
            n/a.

        Returns:
            torch.Tensor: Instance nodes X coordinates.
        """
        return self.env.td_state["coords"][:, :, 0]

    def get_feat_y_coordinate(self):
        """ 
        Instance nodes Y coordinates.

        Args:
            n/a.

        Returns:
            torch.Tensor: Instance nodes Y coordinates.
        """
        return self.env.td_state["coords"][:, :, 1]

    def get_feat_x_coordinate_min_max(self):
        """ 
        Min-max normalized X coordinates of instance nodes.

        Args:
            n/a.

        Returns:
            torch.Tensor: Min. and max. x coordinates of instance nodes.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        feat = ncoord[:,:, 0]
        return feat

    def get_feat_y_coordinate_min_max(self):
        """ 
        Min-max normalized Y coordinates of instance nodes.

        Args:
            n/a.

        Returns:
            torch.Tensor: Min-max normalized Y coordinates of instance nodes.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        feat = ncoord[:, :, 1]
        return feat

    def get_feat_tw_low(self):
        """ 
        Nodes time windows starting times.

        Args:
            n/a.

        Returns:
            torch.Tensor: Nodes time windows starting times.
        """
        #return self.env.td_state['tw_low'] / self.env.td_state['max_tour_duration'].unsqueeze(dim=-1)
        return torch.nan_to_num(self.env.td_state['tw_low'], nan=0.0, posinf=0.0, neginf=0.0)

    def get_feat_tw_high(self):
        """ 
        Nodes time windows ending times.

        Args:
            n/a.

        Returns:
            torch.Tensor: Nodes time windows ending times.
        """
        #return torch.nan_to_num(self.env.td_state['tw_high'], posinf=0.0) / self.env.td_state['max_tour_duration'].unsqueeze(dim=-1)
        return torch.nan_to_num(self.env.td_state['tw_high'], nan=0.0, posinf=0.0, neginf=0.0)

    def get_feat_linehaul_demand(self):
        """
        Nodes linehaul demands.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes linehaul demands.
        """
        return (self.env.td_state['linehaul_demands'] / self.env.td_state['capacity'])
    
    def get_feat_backhaul_demand(self):
        """
        Nodes backhaul demands.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes backhaul demands.
        """
        return (self.env.td_state['backhaul_demands'] / self.env.td_state['capacity'])

    def get_feat_service_time(self):
        """
        Nodes service times.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes service times.
        """
        return self.env.td_state['service_time']

    def get_feat_tw_high_minus_tw_low_div_max_dur(self):
        """ 
        Nodes time window amplitude divided by max tour duration.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes time window amplitude divided by max tour duration.
        """
        tw_high = self.get_feat_tw_high()
        tw_low = self.get_feat_tw_low()
        return (tw_high-tw_low) / self.env.td_state['max_tour_duration'].unsqueeze(dim=-1)

    def get_feat_is_depot(self):
        """ 
        Checks if node is depot.
        
        Args:
            n/a.

        Returns: 
            torch.Tensor: If the node is depot or not.
        """
        return self.env.td_state['is_depot']
        

    ## dynamic features
    def get_feat_time2open_div_end_time(self):
        """ 
        Nodes time to open divided by end time.
        
        Args:
            n/a.

        Returns: 
            torch.Tensor: Nodes time to open divided by end time.
        """
        feat = (self.env.td_state['tw_low'] - self.env.td_state['cur_agent']['cur_time']) / self.env.td_state['end_time'].unsqueeze(dim=-1)
        return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    def get_feat_time2close_div_end_time(self):
        """
        Nodes time to close divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes time to close divided by end time.
        """
        feat = (self.env.td_state['tw_high'] - self.env.td_state['cur_agent']['cur_time']) / self.env.td_state['end_time'].unsqueeze(dim=-1)
        return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    def get_feat_arrive2node_div_end_time(self):
        """ 
        Agent arriving time to nodes divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agent arriving time to nodes divided by end time.
        """
        loc = self.env.td_state['coords'].gather(1, self.env.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        ptime = self.env.td_state['cur_agent']['cur_time'].clone()
        time2j = torch.pairwise_distance(loc, self.env.td_state["coords"], eps=0, keepdim = False)
        arrivej = ptime + time2j
        feat = arrivej / self.env.td_state['end_time'].unsqueeze(dim=-1)
        return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    
    def get_feat_time2open_after_step_div_end_time(self):
        """ 
        Nodes time to open, after agent step, divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes time to open, after agent step, divided by end time.
        """

        arrivej = self.get_feat_arrive2node_div_end_time() * self.env.td_state['end_time'].unsqueeze(dim=-1)
        feat = (self.env.td_state['tw_low'] - arrivej) / self.env.td_state['end_time'].unsqueeze(dim=-1)
        return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0) 
    
    def get_feat_time2close_after_step_div_end_time(self):
        """
        Nodes time to close, after agent step, divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes time to close, after agent step, divided by end time.
        """
        arrivej = self.get_feat_arrive2node_div_end_time() * self.env.td_state['end_time'].unsqueeze(dim=-1)
        feat = (self.env.td_state['tw_high'] - arrivej) / self.env.td_state['end_time'].unsqueeze(dim=-1)
        return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0) 

    def get_feat_time2end_after_step_div_end_time(self):
        """ 
        Time end, after agent step to node, divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Time end, after agent step to node, divided by end time.
        """
        arrivej = self.get_feat_arrive2node_div_end_time() * self.env.td_state['end_time'].unsqueeze(dim=-1)
        feat = (self.env.td_state['end_time'].unsqueeze(dim=-1) - arrivej) / self.env.td_state['end_time'].unsqueeze(dim=-1)
        return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0) 
       
    def get_feat_fract_time_after_step_div_end_time(self):
        """
        Fraction of time left, after agent step to node.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Fraction of time left, after agent step to node.
        """
        arrivej = self.get_feat_arrive2node_div_end_time() * self.env.td_state['end_time'].unsqueeze(dim=-1)
        feat = (arrivej - self.env.td_state['start_time'].unsqueeze(dim=-1))/ self.env.td_state['end_time'].unsqueeze(dim=-1)
        return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0) 
    
    def get_feat_reachable_frac_agents(self):
        """ 
        Feasible nodes per agent.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Feasible nodes per agent.
        """
        feat = self.env.td_state['agents']['action_mask'].sum(dim=1)

        return feat / self.env.num_agents
    
    ## Agent features
    def get_feat_agent_x_coordinate(self):
        """
        Current agent X coordinate.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Current agent X coordinate.
        """
        loc = self.env.td_state["coords"].gather(1, self.env.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 0]
        return feat

    def get_feat_agent_y_coordinate(self):
        """ 
        Current agent Y coordinate.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Current agent Y coordinate.
        """
        loc = self.env.td_state["coords"].gather(1, self.env.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 1]
        return feat
    
    def get_feat_agent_x_coordinate_min_max(self):
        """ 
        Current agent min-max normalized X location.

        Args:
            n/a.

        Returns:
            torch.Tensor: Current agent min-max normalized X location.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        loc = ncoord.gather(1, self.env.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 0]
        return feat

    def get_feat_agent_y_coordinate_min_max(self):
        """ 
        Current agent min-max normalized Y location.

        Args:
            n/a.

        Returns:
            torch.Tensor: Current agent min-max normalized Y location.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        loc = ncoord.gather(1, self.env.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 1]
        return feat

    def get_feat_agent_frac_current_time(self):
        """ 
        Agent fraction of time elapsed.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agent fraction of time elapsed.
        """
        feat =  (self.env.td_state['cur_agent']['cur_time'] - self.env.td_state['start_time'].unsqueeze(1)) 
        return feat / self.env.td_state['max_tour_duration'].unsqueeze(1)
    
    def get_feat_agent_cur_linehaul_load(self):
        """
        Current linehaul load carried by agent.

        Args:
            n/a.

        Returns:
            torch.Tensor: Current linehaul load.
        """
        
        cur_linehaul_load = (self.env.td_state['cur_agent']['cur_linehaul_load'])
        return cur_linehaul_load
    
    def get_feat_agent_cur_backhaul_load(self):
        """
        Current backhaul load carried by agent.

        Args:
            n/a.

        Returns:
            torch.Tensor: Current backhaul load.
        """
        
        cur_backhaul_load = (self.env.td_state['cur_agent']['cur_backhaul_load'])
        return cur_backhaul_load
    
    def get_feat_agent_avaliable_load(self):
        """
        Agent's avaliable load when problem is unmixed.

        Args:
            n/a.
        
        Returns:
            torch.Tensor: Avaliable load.
        """

        #If backhaul class = 1

        used_capacity = torch.where(self.env.td_state['cur_agent']['cur_backhaul_load'] == 0, self.env.td_state['cur_agent']['cur_linehaul_load'], self.env.td_state['cur_agent']['cur_backhaul_load'])
        avaliable_load = self.env.td_state['capacity'] - used_capacity

        return avaliable_load
    
    def get_feat_agent_avaliable_load_vrpmpd(self):
        """
        Agent's avaliable load when problem is mixed.

        Args:
            n/a.
        
        Returns:
            torch.Tensor: Avaliable load.
        """

        #If backhaul class = 2

        avaliable_load = (self.env.td_state['capacity'] - self.env.td_state['cur_agent']['cur_backhaul_load']) * (self.env.td_state['backhaul_class'] == 2)
        return avaliable_load
    
    def get_feat_agent_remaining_dist(self):
        """
        Agent's remaining distance.

        Args:
            n/a.

        Returns:
            torch.Tensor: Remaining distance.
        """

        self.default_remaining_dist = 10

        remaining_dist = self.env.td_state['distance_limits'] - self.env.td_state['cur_agent']['cur_route_length']

        return torch.nan_to_num(remaining_dist, posinf=self.default_remaining_dist)

    def get_feat_agent_arrivedepot_div_end_time(self):
        """ 
        Agent time to depot divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agent time to depot divided by end time.
        """
        loc = self.env.td_state['coords'].gather(1, self.env.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        ptime = self.env.td_state['cur_agent']['cur_time'].clone()
        curr_depot = self.env.td_state['depot_loc'].gather(1, self.env.td_state['cur_agent']['depot_idx'].unsqueeze(-1).expand(-1, -1, 2))
        time2depot = torch.pairwise_distance(loc, curr_depot.squeeze(1), eps=0, keepdim = False)
        arrivej = ptime + time2depot

        feat = (arrivej - self.env.td_state['start_time'].unsqueeze(1))

        return feat / self.env.td_state['end_time'].unsqueeze(1)

    def get_feat_agent_frac_feasible_nodes(self):
        """ 
        Fraction of current agent feasible nodes, in order to the total number of instance nodes.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Fraction of current agent feasible nodes, in order to the total number of instance nodes.
        """
        feat = self.env.td_state['cur_agent']['action_mask'].sum(dim=1).unsqueeze(1)
        return feat / self.env.num_nodes
    
    def get_feat_agent_current_time(self):
        """
        Agent's current time.

        Args:
            n/a.

        Returns:
            torch.Tensor: Current time.
        """

        feat = self.env.td_state['cur_agent']['cur_time']
        return feat

    def get_feat_agents_dist2depot_div_end_time(self):
        """
        Fraction of current agent distance to depot compared to its end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Fraction of current agent distance to depot compared to its end time.
        """
        locs = self.env.td_state["coords"].gather(1, self.env.td_state['agents']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        curr_depot = self.env.td_state['depot_loc'].gather(1, self.env.td_state['cur_agent']['depot_idx'].unsqueeze(-1).expand(-1, -1, 2))
        feat = torch.pairwise_distance(curr_depot.squeeze(1), locs, eps=0, keepdim = False)
        return feat  / self.env.td_state['end_time'].unsqueeze(dim=-1)    

    ## Other agents features
    def get_feat_other_agents_x_coordinate(self):
        """ 
        Agents X coordinates.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agents X coordinates.
        """
        loc = self.env.td_state["coords"].gather(1, self.env.td_state['agents']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 0]
        return feat

    def get_feat_other_agents_y_coordinate(self):
        """ 
        Agents Y coordinates.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agents Y coordinates.
        """
        loc = self.env.td_state["coords"].gather(1, self.env.td_state['agents']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 1]
        return feat

    def get_feat_other_agents_x_coordinate_min_max(self):
        """ 
        Agents min-max normalized X location.

        Args:
            n/a.

        Returns:
            torch.Tensor: Agents min-max normalized X location.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        loc = ncoord.gather(1, self.env.td_state['agents']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 0]
        return feat

    def get_feat_other_agents_y_coordinate_min_max(self):
        """
        Agents min-max normalized Y location.

        Args:
            n/a.

        Returns:
            torch.Tensor: Agents min-max normalized Y location.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        loc = ncoord.gather(1, self.env.td_state['agents']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 1]
        return feat

    def get_feat_other_agents_frac_current_time(self):
        """ 
        Agents fraction of elapsed time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agents fraction of elapsed time.
        """
        feats = self.env.td_state['agents']['cur_time'] / self.env.td_state['end_time'].unsqueeze(dim=-1)
        return feats


    def get_feat_other_agents_frac_feasible_nodes(self):
        """ 
        Fraction of agents feasible nodes, in order to the total number of instance nodes.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Fraction of agents feasible nodes, in order to the total number of instance nodes.
        """
        feat = self.env.td_state['agents']['action_mask'].sum(dim=-1)
        return feat / self.env.num_nodes

    def get_feat_other_agents_dist2agent_div_end_time(self):
        """ 
        Agents distance to active agent divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agents distance to active agent divided by end time.
        """
        locs = self.env.td_state["coords"].gather(1, self.env.td_state['agents']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        loc = self.env.td_state['coords'].gather(1, self.env.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        
        feat = torch.pairwise_distance(loc, locs, eps=0, keepdim = False)
        return feat  / self.env.td_state['end_time'].unsqueeze(dim=-1)

    def get_feat_other_agents_time_delta2agent_div_max_dur(self):
        """ 
        Difference between agents time and current agent time, divided by max. tour duration.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Difference between agents time and current agent time, divided by max. tour duration.
        """
        feats = (self.env.td_state['agents']['cur_time'] - self.env.td_state['cur_agent']['cur_time'] )/ self.env.td_state['max_tour_duration'].unsqueeze(dim=-1)
        return feats  

    def get_feat_other_agents_was_last(self):
        """
        Last agent performing an action.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Last agent performing an action.
        """
        feats = torch.zeros_like(self.env.td_state['agents']['active_agents_mask'], dtype=torch.long).scatter_(1, self.env.td_state['cur_agent_idx'], torch.ones_like(self.env.td_state['cur_agent_idx']))
        return feats   

    ## All agents features
    def get_feat_all_agents_x_coordinate(self):
        """ 
        Agents X coordinates.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agents X coordinates.
        """
        loc = self.env.td_state["coords"].gather(1, self.env.td_state['agents']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 0]
        return feat

    def get_feat_all_agents_y_coordinate(self):
        """ 
        Agents Y coordinates.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agents Y coordinates.
        """
        loc = self.env.td_state["coords"].gather(1, self.env.td_state['agents']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 1]
        return feat

    ## Global features

    def get_feat_global_frac_done_agents(self):
        """
        Fraction of done agents.

        Args:
            n/a.

        Returns: 
            torch.Tensor: Fraction of done agents.
        """
        feat = self.env.td_state['agents']['active_agents_mask'].sum(dim=1).unsqueeze(1)
        return 1 - (feat / self.env.num_agents)

    def get_feat_global_frac_linehaul_demands(self):
        """
        Fraction of served demands.

        Args:
            n/a.

        Returns: 
            torch.Tensor: Fraction of served demands.
        """
        feat = self.env.td_state['nodes']['linehaul_demands'].sum(dim=-1).unsqueeze(1)
        return feat / self.env.td_state['linehaul_demands'].sum(dim=-1).unsqueeze(1)
    
    def get_feat_global_frac_backhaul_demands(self):
        """
        Fraction of served demands.

        Args:
            n/a.

        Returns: 
            torch.Tensor: Fraction of served demands.
        """
        feat = self.env.td_state['nodes']['backhaul_demands'].sum(dim=-1).unsqueeze(1)
        return feat / self.env.td_state['backhaul_demands'].sum(dim=-1).unsqueeze(1)
    
    def get_feat_global_open_routes(self):

        """
        Checks if problems have open routes.

        Args:
            n/a.

        Returns:
            torch.Tensor: Open routes.
        """

        open_routes = self.env.td_state['open_routes']
        return open_routes
    
    def get_feat_global_distance_limits(self):

        """
        Chack problems distance limits.

        Args:
            n/a.

        Returns:
            torch.Tensor: Distance limits.
        """

        return torch.nan_to_num(self.env.td_state['distance_limits'], posinf=0.0)
    
    def get_feat_global_max_tw_depot(self):

        """
        High tw from depot. Max tour duration.

        Args:
            N/a.

        Returns:
            torch.Tensor: Max tour duration.
        """

        return torch.nan_to_num(self.env.td_state['max_tour_duration'].unsqueeze(-1), posinf=0.0)
    
    def get_feat_global_is_problem_backhaul_mixed(self):

        """
        Checks if problem is mixed. 0 means it's unmixed, 1 means it's mixed.

        Args:
            n/a.

        Returns:
            torch.Tensor: Is problem mixed?
        """

        return (self.env.td_state['backhaul_class']==2).float()
    
    # --------------------------------------------------------------------------------------