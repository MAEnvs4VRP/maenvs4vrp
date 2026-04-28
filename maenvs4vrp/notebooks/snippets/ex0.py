print('cum profit:', env.td_state['cur_agent']['step_profit'])
loc = env.td_state['coords'].gather(1, env.td_state['cur_agent']['cur_node_idx'][:,:,None].expand(-1, -1, 2))
time2j = torch.pairwise_distance(loc, env.td_state["coords"], eps=0, keepdim = False)
print('dist:', time2j[0])
print('action_mask:', env.td_state['cur_agent']['action_mask'])

