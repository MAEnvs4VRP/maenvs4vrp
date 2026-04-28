td['next_action'] = torch.tensor([[3]])
td = env.step_agent_select_observe(td)
print("reward: ", td['reward'])
print("penalty: ", td['penalty'])