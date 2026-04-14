import torch
from tensordict import TensorDict

from maenvs4vrp.core.env_observation_builder import ObservationBuilder
from maenvs4vrp.core.env import AECEnv

from typing import Optional, Dict


class Observations(ObservationBuilder):
    """
    CVRPTW observations class.
    """

    POSSIBLE_NODES_STATIC_FEATURES = ['x_coordinate', 'y_coordinate', 'tw_low',
                                    'tw_high', 'demand', 'service_time', 'tw_high_minus_tw_low_div_max_dur',
                                    'x_coordinate_min_max', 'y_coordinate_min_max', 'is_depot']

    POSSIBLE_NODES_DYNAMIC_FEATURES = ['time2open_div_end_time', 'time2close_div_end_time', 'arrive2node_div_end_time',
                                'time2open_after_step_div_end_time', 'time2close_after_step_div_end_time',
                                'time2end_after_step_div_end_time', 'fract_time_after_step_div_end_time',
                                'reachable_frac_agents']

    POSSIBLE_AGENT_FEATURES = ['x_coordinate', 'y_coordinate','x_coordinate_min_max', 'y_coordinate_min_max', 'frac_current_time', 
                                'frac_current_load', 'arrivedepot_div_end_time', 
                                'frac_feasible_nodes']

    POSSIBLE_OTHER_AGENTS_FEATURES = ['x_coordinate', 'y_coordinate','x_coordinate_min_max', 'y_coordinate_min_max', 'frac_current_time', 
                                    'frac_current_load', 'dist2depot_div_end_time', 
                                    'dist2agent_div_end_time', 'frac_feasible_nodes','time_delta2agent_div_max_dur', 'was_last']

    POSSIBLE_GLOBAL_FEATURES = ['frac_fleet_load_capacity', 'frac_done_agents']


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
                                    'demand': {'feat': 'demand', 'norm': None},
                                    'is_depot': {'feat': 'is_depot', 'norm': None}},
                                    'nodes_dynamic': [],
                                    'agent': ['frac_current_load'],
                                    'other_agents': [],
                                    'all_agents': [],
                                    'global': [ 'frac_fleet_load_capacity', 'frac_done_agents']}

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
        return self.env.td_state['tw_low'] / self.env.td_state['max_tour_duration'].unsqueeze(dim=-1)

    def get_feat_tw_high(self):
        """ 
        Nodes time windows ending times.

        Args:
            n/a.

        Returns:
            torch.Tensor: Nodes time windows ending times.
        """
        return self.env.td_state['tw_high'] / self.env.td_state['max_tour_duration'].unsqueeze(dim=-1)

    def get_feat_demand(self):
        """
        Nodes demand.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes demand.
        """
        return self.env.td_state['demands'] / self.env.td_state['capacity']

    def get_feat_service_time(self):
        """
        Nodes service time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes service time.
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
        return feat

    def get_feat_time2close_div_end_time(self):
        """
        Nodes time to close divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes time to close divided by end time.
        """
        feat = (self.env.td_state['tw_high'] - self.env.td_state['cur_agent']['cur_time']) / self.env.td_state['end_time'].unsqueeze(dim=-1)
        return feat

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
        return arrivej / self.env.td_state['end_time'].unsqueeze(dim=-1)
    
    def get_feat_time2open_after_step_div_end_time(self):
        """ 
        Nodes time to open, after agent step, divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes time to open, after agent step, divided by end time.
        """

        arrivej = self.get_feat_arrive2node_div_end_time() * self.env.td_state['end_time'].unsqueeze(dim=-1)
        feat = (self.env.td_state['tw_low'] - arrivej) 
        return feat / self.env.td_state['end_time'].unsqueeze(dim=-1)
    
    def get_feat_time2close_after_step_div_end_time(self):
        """
        Nodes time to close, after agent step, divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Nodes time to close, after agent step, divided by end time.
        """
        arrivej = self.get_feat_arrive2node_div_end_time() * self.env.td_state['end_time'].unsqueeze(dim=-1)
        feat = (self.env.td_state['tw_high'] - arrivej) 
        return feat / self.env.td_state['end_time'].unsqueeze(dim=-1)

    def get_feat_time2end_after_step_div_end_time(self):
        """ 
        Time end, after agent step to node, divided by end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Time end, after agent step to node, divided by end time.
        """
        arrivej = self.get_feat_arrive2node_div_end_time() * self.env.td_state['end_time'].unsqueeze(dim=-1)
        feat = (self.env.td_state['end_time'].unsqueeze(dim=-1) - arrivej)
        return feat / self.env.td_state['end_time'].unsqueeze(dim=-1)
       
    def get_feat_fract_time_after_step_div_end_time(self):
        """
        Fraction of time left, after agent step to node.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Fraction of time left, after agent step to node.
        """
        arrivej = self.get_feat_arrive2node_div_end_time() * self.env.td_state['end_time'].unsqueeze(dim=-1)
        feat = (arrivej - self.env.td_state['start_time'].unsqueeze(dim=-1))
        return feat / self.env.td_state['end_time'].unsqueeze(dim=-1)
    
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

    def get_feat_agent_frac_current_load(self):
        """
        Agent fraction of used capacity.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agent fraction of used capacity.
        """
        feat =  (self.env.td_state['cur_agent']['cur_load'] - self.env.td_state['agents']['capacity'])  / self.env.td_state['agents']['capacity']
        return feat

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
        time2depot = torch.pairwise_distance(loc, self.env.td_state['depot_loc'], eps=0, keepdim = False)
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

    def get_feat_agents_dist2depot_div_end_time(self):
        """
        Fraction of current agent distance to depot compared to its end time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Fraction of current agent distance to depot compared to its end time.
        """
        locs = self.env.td_state["coords"].gather(1, self.env.td_state['agents']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = torch.pairwise_distance(self.env.td_state['depot_loc'], locs, eps=0, keepdim = False)
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
    
    def get_feat_agents_y_coordinate_min_max(self):
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

    def get_feat_other_agents_frac_current_load(self):
        """ 
        Agents fraction of used capacity.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agents fraction of used capacity.
        """
        feats = self.env.td_state['agents']['cur_load'] / self.env.td_state['agents']['capacity']
        return feats

    def get_feat_other_agents_remaining_capacity(self):
        """
        Agent remaining capacity.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agent fraction of used capacity.
        """
        feat =  self.env.td_state['agents']['cur_load']
        return feat

    def get_feat_agents_frac_feasible_nodes(self):
        """ 
        Fraction of agents feasible nodes, in order to the total number of instance nodes.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Fraction of agents feasible nodes, in order to the total number of instance nodes.
        """
        feat = self.env.td_state['agents']['action_mask'].sum(dim=-1)
        return feat / self.env.num_nodes
    
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

    ## All agents
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

    def get_feat_all_agents_cur_time(self):
        """ 
        Agents current time.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agents current time.
        """
        feats = self.env.td_state['agents']['cur_time'] 
        return feats

    def get_feat_all_agents_remaining_capacity(self):
        """
        Agent remaining capacity.

        Args:
            n/a.
        
        Returns: 
            torch.Tensor: Agent fraction of used capacity.
        """
        feat =  self.env.td_state['agents']['cur_load']
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


    def get_feat_global_frac_fleet_load_capacity(self):
        """
        Fraction of fleet load capacity.

        Args:
            n/a.

        Returns: 
            torch.Tensor: Fraction of fleet load capacity.
        """
        feat = self.env.td_state['agents']['cur_load'].sum(dim=-1).unsqueeze(1)
        capacity = self.env.td_state['agents']['capacity']
        return feat / (capacity * self.env.num_agents)
    
    # --------------------------------------------------------------------------------------