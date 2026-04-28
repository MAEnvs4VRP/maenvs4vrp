class Observations(Observations):
    
    def __init__(self, feature_list:dict = None):
        super().__init__()
        
        self.default_feature_list['nodes_dynamic'].append('wait_time_div_end_time')
        self.possible_nodes_dynamic_features.append('wait_time_div_end_time')
    
    def get_feat_wait_time_div_end_time(self):
        """ dynamic feature
        Args:

        Returns: 
            Tensor: waiting time at nodes divided by end time.
        """
        loc = self.env.td_state['coords'].gather(1, self.env.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        ptime = self.env.td_state['cur_agent']['cur_time'].clone()
        time2j = torch.pairwise_distance(loc, self.env.td_state["coords"], eps=0, keepdim = False)
        arrivej = ptime + time2j
        wait = torch.clip(self.env.td_state['tw_low'] - arrivej, min=0)
        return wait / self.env.td_state['end_time'].unsqueeze(dim=-1)
    