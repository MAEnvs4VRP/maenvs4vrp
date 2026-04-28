class Environment(Environment):

    def _update_state(self, action):
        loc = self.td_state['coords'].gather(1, self.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
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
        self.td_state['cur_agent']['cur_node_idx'] = action
        self.td_state['agents']['cur_node_idx'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_node_idx'])
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
        cur_demands = self.td_state['nodes']['cur_demands'].gather(1, action)
        current_load = self.td_state['cur_agent']['cur_load']
        load_transfer = torch.minimum(cur_demands, current_load)

        self.td_state['cur_agent']['cur_load'] -= load_transfer

        # if agent is done set agent cur_load to 0
        self.td_state['cur_agent']['cur_load'] = torch.where(agents_done, 0., 
                                                             self.td_state['cur_agent']['cur_load'])
        
        self.td_state['nodes']['cur_demands'].scatter_(1, action, cur_demands-load_transfer)
        # update done nodes
        self.td_state['nodes']['active_nodes_mask'] = self.td_state['nodes']['cur_demands'].gt(0)
        self.td_state['nodes']['active_nodes_mask'].scatter_(1, self.td_state['depot_idx'], True)

        self.td_state['agents']['cur_load'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_load'])
        # update visited nodes
        r = torch.arange(*self.td_state.batch_size, device=self.device)
        self.td_state['agents']['visited_nodes'][r, self.td_state['cur_agent_idx'].squeeze(-1), action.squeeze(-1)] = True
        # update agent step
        self.td_state['cur_agent']['cur_step'] = torch.where(~agents_done, self.td_state['cur_agent']['cur_step']+1, 
                                                             self.td_state['cur_agent']['cur_step'])
        self.td_state['agents']['cur_step'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_step'])

        # if all done activate first agent to guarantee batch consistency during agent sampling
        self.td_state['agents']['active_agents_mask'][self.td_state['agents']['active_agents_mask'].sum(1).eq(0), 0] = True