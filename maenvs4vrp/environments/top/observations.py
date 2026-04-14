import torch
from tensordict import TensorDict

from maenvs4vrp.core.env_observation_builder import ObservationBuilder
from maenvs4vrp.core.env import AECEnv

from typing import Optional, Dict


class Observations(ObservationBuilder):
    """Observations class

    Every featute on POSSIBLE_NODES_STATIC_FEATURES, POSSIBLE_NODES_DYNAMIC_FEATURES, POSSIBLE_SELF_FEATURES
    POSSIBLE_OTHER_AGENTS_FEATURES and POSSIBLE_GLOBAL_FEATURES list have to have their corresponding method.

    Ex: 
    'x_coordinate' -> get_feat_x_coordinate
    'time2open_div_end_time' -> get_feat_time2open_div_end_time
    
    """

    POSSIBLE_NODES_STATIC_FEATURES = ['x_coordinate', 'y_coordinate', 'profits', 'service_time', 
                                      'x_coordinate_min_max', 'y_coordinate_min_max', 'is_depot']

    POSSIBLE_NODES_DYNAMIC_FEATURES = [ 'arrive2node_div_end_time', 'fract_time_after_step_div_end_time', 'reachable_frac_agents']

    POSSIBLE_AGENT_FEATURES = ['x_coordinate', 'y_coordinate', 'x_coordinate_min_max', 'y_coordinate_min_max', 'frac_current_time', 
                                'frac_current_profit', 'arrivedepot_div_end_time', 'frac_feasible_nodes']

    POSSIBLE_OTHER_AGENTS_FEATURES = ['x_coordinate', 'y_coordinate', 'x_coordinate_min_max', 'y_coordinate_min_max', 'frac_current_time', 
                                'frac_current_profit', 'dist2agent_div_end_time', 'frac_feasible_nodes', 'frac_time_left', 
                                'time_delta2agent_div_max_dur', 'was_last']
    
    POSSIBLE_ALL_AGENTS_FEATURES = ['x_coordinate', 'y_coordinate','x_coordinate_min_max', 'y_coordinate_min_max',  
                                    'remaining_capacity', 'was_last']
    
    POSSIBLE_GLOBAL_FEATURES = ['frac_profits', 'frac_colect_profits', 'frac_done_agents']


    def __init__(self, feature_list:Dict = None):
        super().__init__()
        """
        Args:
            feature_list (Dict): dictionary containing observation features list to be available to the agent;

        """

        self.default_feature_list = {'nodes_static': {'x_coordinate': {'feat': 'x_coordinate', 'norm': None},
                                    'y_coordinate': {'feat': 'y_coordinate', 'norm': None},
                                    'profits': {'feat': 'profits', 'norm': None},
                                    'is_depot': {'feat': 'is_depot', 'norm': None}},
                                    'nodes_dynamic': [],
                                    'agent': ['frac_current_profit'],
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

    
    ## static features
    def get_feat_x_coordinate(self):
        """ static feature
        Args:

        Returns: 
            npt.NDArray: x coordinates of instance nodes.
        """
        return self.env.td_state["coords"][:, :, 0]

    def get_feat_y_coordinate(self):
        """ static feature
        Args:

        Returns: 
            npt.NDArray: y coordinates of instance nodes.
        """
        return self.env.td_state["coords"][:, :, 1]

    def get_feat_x_coordinate_min_max(self):
        """ static feature
        Args:

        Returns: 
            npt.NDArray: min-max normalized x coordinates of instance nodes.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        feat = ncoord[:,:, 0]
        return feat

    def get_feat_y_coordinate_min_max(self):
        """ static feature
        Args:

        Returns:
            npt.NDArray: min-max normalized y coordinates of instance nodes.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        feat = ncoord[:, :, 1]
        return feat

    def get_feat_profits(self):
        """ static feature
        Args:

        Returns: 
            npt.NDArray: nodes' profits.
        """
        return self.env.td_state['profits']

    def get_feat_service_time(self):
        """ static feature
        Args:

        Returns: 
            npt.NDArray: nodes' service time.
        """
        return self.env.td_state['service_time']


    def get_feat_is_depot(self):
        """ static feature
        Args:

        Returns: 
            npt.NDArray: is depot bool
        """
        return self.env.td_state['is_depot']
        

    ## dynamic features

    def get_feat_arrive2node_div_end_time(self):
        """ dynamic feature
        Args:

        Returns: 
            npt.NDArray: agent arrive time to nodes divided by end time.
        """
        loc = self.env.td_state['coords'].gather(1, self.env.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        ptime = self.env.td_state['cur_agent']['cur_time'].clone()
        time2j = torch.pairwise_distance(loc, self.env.td_state["coords"], eps=0, keepdim = False)
        arrivej = ptime + time2j
        return arrivej / self.env.td_state['end_time'].unsqueeze(dim=-1)
    

    def get_feat_time2end_after_step_div_end_time(self):
        """ dynamic feature
        Args:

        Returns: 
            npt.NDArray: time end, after agent step to node, divided by end time.
        """
        arrivej = self.get_feat_arrive2node_div_end_time() * self.env.td_state['end_time'].unsqueeze(dim=-1)
        feat = (self.env.td_state['end_time'].unsqueeze(dim=-1) - arrivej)
        return feat / self.env.td_state['end_time'].unsqueeze(dim=-1)
       
    def get_feat_fract_time_after_step_div_end_time(self):
        """ dynamic feature
        Args:

        Returns: 
            npt.NDArray: fraction of time left, after agent step to node.
        """
        arrivej = self.get_feat_arrive2node_div_end_time() * self.env.td_state['end_time'].unsqueeze(dim=-1)
        feat = (arrivej - self.env.td_state['start_time'].unsqueeze(dim=-1))
        return feat / self.env.td_state['end_time'].unsqueeze(dim=-1)
    

    def get_feat_reachable_frac_agents(self):
        """ dynamic feature
        Args:

        Returns: 
            npt.NDArray: fraction of time left, after agent step to node.
        """
        feat = self.env.td_state['agents']['action_mask'].sum(dim=1)

        return feat / self.env.num_agents
    
    ## Agent features
    def get_feat_agent_x_coordinate(self):
        """ active agent feature
        Args:

        Returns: 
            int: agent current x location.
        """
        loc = self.env.td_state["coords"].gather(1, self.env.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 0]
        return feat

    def get_feat_agent_y_coordinate(self):
        """ active agent feature
        Args:

        Returns 
            int: agent current y location.
        """
        loc = self.env.td_state["coords"].gather(1, self.env.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 1]
        return feat
    
    def get_feat_agent_x_coordinate_min_max(self):
        """ active agent feature
        Args:

        Returns: 
            int: agent current min-max normalized x location.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        loc = ncoord.gather(1, self.env.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 0]
        return feat

    def get_feat_agent_y_coordinate_min_max(self):
        """ active agent feature
        Args:

        Returns 
            int: agent current min-max normalized y location.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        loc = ncoord.gather(1, self.env.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 1]
        return feat

    def get_feat_agent_frac_current_time(self):
        """ active agent feature
        Args:

        Returns: 
            int: agent fraction of time elapsed.
        """
        feat =  (self.env.td_state['cur_agent']['cur_time'] - self.env.td_state['start_time'].unsqueeze(1)) 
        return feat / self.env.td_state['max_tour_duration'].unsqueeze(1)

    def get_feat_agent_frac_current_profit(self):
        """ active agent feature
        Args:

        Returns: 
            int: agent fraction of cum profit.
        """
        feat =  self.env.td_state['cur_agent']['cum_profit'] / self.env.td_state['profits'].sum(dim=-1).unsqueeze(1)
        return feat

    def get_feat_agent_arrivedepot_div_end_time(self):
        """ active agent feature
        Args:

        Returns: 
            int: agent time to depot divided by end time.
        """
        loc = self.env.td_state['coords'].gather(1, self.env.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        ptime = self.env.td_state['cur_agent']['cur_time'].clone()
        time2depot = torch.pairwise_distance(loc, self.env.td_state['depot_loc'], eps=0, keepdim = False)
        arrivej = ptime + time2depot

        feat = (arrivej - self.env.td_state['start_time'].unsqueeze(1))

        return feat / self.env.td_state['end_time'].unsqueeze(1)

    def get_feat_agent_frac_feasible_nodes(self):
        """ active agent feature
        Args:

        Returns: 
            int: fraction of feasible nodes, in order to the total number of instance nodes.
        """
        feat = self.env.td_state['cur_agent']['action_mask'].sum(dim=1).unsqueeze(1)
        return feat / self.env.num_nodes

    

    ## Other other_agents features
    def get_feat_other_agents_x_coordinate(self):
        """ active agent feature
        Args:

        Returns: 
            int: agent current min-max normalized x location.
        """
        loc = self.env.td_state["coords"].gather(1, self.env.td_state['agents']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 0]
        return feat
    
    def get_feat_other_agents_y_coordinate(self):
        """ active agent feature
        Args:

        Returns: 
            int: agent current min-max normalized y location.
        """
        loc = self.env.td_state["coords"].gather(1, self.env.td_state['agents']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 1]
        return feat
    

    
    def get_feat_other_agents_frac_current_time(self):
        """ agents features
        Args:

        Returns: 
            npt.NDArray: agents fraction of elapsed time.
        """
        feats = self.env.td_state['agents']['cur_time'] / self.env.td_state['end_time'].unsqueeze(dim=-1)
        return feats
    
    def get_feat_other_agents_time_delta2agent_div_max_dur(self):
        """ agents features
        Args:

        Returns: 
            npt.NDArray: agents fraction of elapsed time.
        """
        feats = (self.env.td_state['agents']['cur_time'] - self.env.td_state['cur_agent']['cur_time'] )/ self.env.td_state['max_tour_duration'].unsqueeze(dim=-1)
        return feats  
    
    def get_feat_other_agents_frac_time_left(self):
        """ agents features
        Args:

        Returns: 
            npt.NDArray: agents fraction of elapsed time.
        """
        feats = 1 - self.env.td_state['agents']['cur_time'] / self.env.td_state['end_time'].unsqueeze(dim=-1)
        return feats
    
    def get_feat_other_agents_frac_current_profit(self):
        """ agents features
        Args:

        Returns: 
            npt.NDArray: agents fraction of cum profit
        """
        feats = self.env.td_state['agents']['cum_profit'] / self.env.td_state['profits'].sum(dim=-1).unsqueeze(1)
        return feats


    def get_feat_other_agents_was_last(self):
        """ agents features
        Args:

        Returns: 
            npt.NDArray: agents fraction of cum profit
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
    
    ## Global features
    def get_feat_global_frac_done_agents(self):
        """global features

        Args:

        Returns: 
            int: fraction of done agents.
        """
        feat = self.env.td_state['agents']['active_agents_mask'].sum(dim=1).unsqueeze(1)
        return 1 - (feat / self.env.num_agents)

    def get_feat_global_frac_profits(self):
        """global features

        Args:

        Returns: 
            int: fraction of remaining profits
        """
        feat = self.env.td_state['nodes']['cur_profits'].sum(dim=-1).unsqueeze(1)
        return feat / self.env.td_state['profits'].sum(dim=-1).unsqueeze(1)

    def get_feat_global_frac_colect_profits(self):
        """global features

        Args:

        Returns: 
            int: fraction of fleet colect profits
        """
        feat = self.env.td_state['agents']['cum_profit'].sum(dim=-1).unsqueeze(1)
        return feat /  self.env.td_state['profits'].sum(dim=-1).unsqueeze(1)
    
    # --------------------------------------------------------------------------------------