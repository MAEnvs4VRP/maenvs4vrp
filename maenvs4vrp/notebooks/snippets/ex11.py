class Environment(Environment):
 
    def _update_feasibility(self):

        _mask = self.td_state['nodes']['active_nodes_mask'].clone() * self.td_state['cur_agent']['action_mask'].clone()

        # time windows constraints
        loc = self.td_state['coords'].gather(1, self.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
        ptime = self.td_state['cur_agent']['cur_time'].clone()
        time2j = torch.pairwise_distance(loc, self.td_state["coords"], eps=0, keepdim = False)
        if self.n_digits is not None:
            time2j = torch.floor(self.n_digits * time2j) / self.n_digits

        arrivej = ptime + time2j
        waitj = torch.clip(self.td_state['tw_low_limit']-arrivej, min=0)
        service_startj = arrivej + waitj

        c0 = arrivej > self.td_state['arrive_limit'] # agents can only arrive at each customer after $o_i - P_{max} - W_{max}$
        c1 = service_startj <= self.td_state['tw_high_limit']
        c2 = service_startj + self.td_state['service_time'] + self.td_state['time2depot'] <= self.td_state['end_time'].unsqueeze(-1)

        # capacity constraints
        c3 = self.td_state['demands'] <= self.td_state['cur_agent']['cur_load']

        _mask = _mask * c0 * c1 * c2 * c3
        # update state
        self.td_state['cur_agent'].update({'action_mask': _mask}) 
        self.td_state['agents']['feasible_nodes'].scatter_(1, 
                                            self.td_state['cur_agent_idx'][:,:,None].expand(-1,-1,self.num_nodes), _mask.unsqueeze(1))
