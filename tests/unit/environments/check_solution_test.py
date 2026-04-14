import pytest
import importlib
import torch

from maenvs4vrp.core import env

ENVIRONMENT_LIST = ['cvrp', 'hcvrp', 'top', 'cvrpstw', 'cvrptw', 'toptw', 'dvrptw', 'dsvrptw', 'mdvrptw', 'pdptw', 'sdvrptw', 'pcvrptw']

DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_AGENTS = [20, 30, 50]
DEFAULT_NUM_NODES = [51, 101, 501, 1001]

@pytest.fixture
def define_values_fixture(device, batch, num_agents, num_nodes):
    defined_device = torch.device(DEFAULT_DEVICE) if device == "cpu" else torch.device("cuda")
    defined_batch = DEFAULT_BATCH_SIZE if batch == None else batch
    defined_num_agents = DEFAULT_NUM_AGENTS if num_agents == None else num_agents
    defined_num_nodes = DEFAULT_NUM_NODES if num_nodes == None else num_nodes

    #Convert every list element from str to int
    defined_batch = int(defined_batch)
    defined_num_agents = list(map(int, defined_num_agents))
    defined_num_nodes = list(map(int, defined_num_nodes))

    return defined_device, defined_batch, defined_num_agents, defined_num_nodes

@pytest.fixture(params=ENVIRONMENT_LIST)
def environment_instances_fixture(request):
    env_agent_selector_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_selector'
    env_agent_selector = importlib.import_module(env_agent_selector_module_name).AgentSelector()

    observations_module_name = f'maenvs4vrp.environments.{request.param}.observations'
    observations = importlib.import_module(observations_module_name).Observations()

    generator_module_name = f'maenvs4vrp.environments.{request.param}.instances_generator'
    generator = importlib.import_module(generator_module_name).InstanceGenerator()

    environment_module_name = f'maenvs4vrp.environments.{request.param}.env'
    environment_module = importlib.import_module(environment_module_name)

    env_agent_reward_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_reward'
    reward_evaluator = importlib.import_module(env_agent_reward_module_name).DenseReward()

    environment = environment_module.Environment(instance_generator_object=generator,
                                                 obs_builder_object=observations,
                                                 agent_selector_object=env_agent_selector,
                                                 reward_evaluator=reward_evaluator,
                                                 )
    return environment

@pytest.fixture(params=ENVIRONMENT_LIST)
def environment_instances_fixture_st(request):
    env_agent_selector_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_selector'
    env_agent_selector = importlib.import_module(env_agent_selector_module_name).SmallestTimeAgentSelector()

    observations_module_name = f'maenvs4vrp.environments.{request.param}.observations'
    observations = importlib.import_module(observations_module_name).Observations()

    generator_module_name = f'maenvs4vrp.environments.{request.param}.instances_generator'
    generator = importlib.import_module(generator_module_name).InstanceGenerator()

    environment_module_name = f'maenvs4vrp.environments.{request.param}.env'
    environment_module = importlib.import_module(environment_module_name)

    env_agent_reward_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_reward'
    reward_evaluator = importlib.import_module(env_agent_reward_module_name).DenseReward()

    environment = environment_module.Environment(instance_generator_object=generator,
                                                 obs_builder_object=observations,
                                                 agent_selector_object=env_agent_selector,
                                                 reward_evaluator=reward_evaluator,
                                                 )
    return environment


@pytest.fixture(params=ENVIRONMENT_LIST)
def environment_instances_fixture_rand(request):
    env_agent_selector_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_selector'
    env_agent_selector = importlib.import_module(env_agent_selector_module_name).RandomSelector()

    observations_module_name = f'maenvs4vrp.environments.{request.param}.observations'
    observations = importlib.import_module(observations_module_name).Observations()

    generator_module_name = f'maenvs4vrp.environments.{request.param}.instances_generator'
    generator = importlib.import_module(generator_module_name).InstanceGenerator()

    environment_module_name = f'maenvs4vrp.environments.{request.param}.env'
    environment_module = importlib.import_module(environment_module_name)

    env_agent_reward_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_reward'
    reward_evaluator = importlib.import_module(env_agent_reward_module_name).DenseReward()

    environment = environment_module.Environment(instance_generator_object=generator,
                                                 obs_builder_object=observations,
                                                 agent_selector_object=env_agent_selector,
                                                 reward_evaluator=reward_evaluator,
                                                 )
    return environment

@pytest.fixture(params=ENVIRONMENT_LIST)
def environment_instances_fixture_no_selection(request):

    observations_module_name = f'maenvs4vrp.environments.{request.param}.observations'
    observations = importlib.import_module(observations_module_name).Observations()

    generator_module_name = f'maenvs4vrp.environments.{request.param}.instances_generator'
    generator = importlib.import_module(generator_module_name).InstanceGenerator()

    environment_module_name = f'maenvs4vrp.environments.{request.param}.env'
    environment_module = importlib.import_module(environment_module_name)

    env_agent_reward_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_reward'
    reward_evaluator = importlib.import_module(env_agent_reward_module_name).DenseReward()

    environment = environment_module.Environment(instance_generator_object=generator,
                                                 obs_builder_object=observations,
                                                 agent_selector_object=None,
                                                 reward_evaluator=reward_evaluator,
                                                 )
    return environment

def test_print_defined_values(define_values_fixture):
    defined_device, defined_batch_size, defined_num_agents, defined_num_nodes = define_values_fixture
    print(f"""
Defined device: {defined_device}
Defined batch size: {defined_batch_size}
Defined num_agents: {defined_num_agents}
Defined num_nodes: {defined_num_nodes}
        """)

# agent selection through rules tests
def test_instance_env_agent_iterator_gives_no_error(environment_instances_fixture, define_values_fixture):
    defined_device, defined_batch_size, defined_num_agents, defined_num_nodes = define_values_fixture
    env = environment_instances_fixture
    for nnodes in defined_num_nodes:
        for nagents in defined_num_agents:
            td = env.reset_agent_select_observe(num_agents=nagents, num_nodes=nnodes, device=defined_device, batch_size=defined_batch_size)
            while not td["done"].all():  
                td = env.sample_action(td)
                td = env.step_agent_select_observe(td)
            env.check_solution_validity()


def test_instance_env_agent_smallesttime_iterator_gives_no_error(environment_instances_fixture_st, define_values_fixture):
    defined_device, defined_batch_size, defined_num_agents, defined_num_nodes = define_values_fixture
    env = environment_instances_fixture_st
    for nnodes in defined_num_nodes:
        for nagents in defined_num_agents:
            td = env.reset_agent_select_observe(num_agents=nagents, num_nodes=nnodes, device=defined_device, batch_size=defined_batch_size)
            while not td["done"].all():  
                td = env.sample_action(td)
                td = env.step_agent_select_observe(td)
            env.check_solution_validity()

def test_instance_env_agent_rand_iterator_gives_no_error(environment_instances_fixture_rand, define_values_fixture):
    defined_device, defined_batch_size, defined_num_agents, defined_num_nodes = define_values_fixture
    env = environment_instances_fixture_rand
    for nnodes in defined_num_nodes:
        for nagents in defined_num_agents:
            td = env.reset_agent_select_observe(num_agents=nagents, num_nodes=nnodes, device=defined_device, batch_size=defined_batch_size)
            while not td["done"].all():  
                td = env.sample_action(td)
                td = env.step_agent_select_observe(td)
            env.check_solution_validity()

# sequential agent-node selection tests
def test_instance_env_agent_node_selection_gives_no_error(environment_instances_fixture_no_selection, define_values_fixture):
    defined_device, defined_batch_size, defined_num_agents, defined_num_nodes = define_values_fixture
    env = environment_instances_fixture_no_selection
    for nnodes in defined_num_nodes:
        for nagents in defined_num_agents:
            td = env.reset(num_agents=nagents, num_nodes=nnodes, device=defined_device, batch_size=defined_batch_size)
            while not td["done"].all():  
                td = env.sample_agent(td)
                td = env.sample_action(td)
                td = env.step_observe(td)
            env.check_solution_validity()

# simultaneous agent-node selection tests
def test_instance_env_joint_selection_gives_no_error(environment_instances_fixture_no_selection, define_values_fixture):
    defined_device, defined_batch_size, defined_num_agents, defined_num_nodes = define_values_fixture
    env = environment_instances_fixture_no_selection
    for nnodes in defined_num_nodes:
        for nagents in defined_num_agents:
            td = env.reset_observe(num_agents=nagents, num_nodes=nnodes, device=defined_device, batch_size=defined_batch_size)
            while not td["done"].all():  
                td = env.sample_joint(td)
                td = env.step_observe(td)
            env.check_solution_validity()

# sequential node-agent selection tests
def test_instance_env_node_agent_selection_gives_no_error(environment_instances_fixture_no_selection, define_values_fixture):
    defined_device, defined_batch_size, defined_num_agents, defined_num_nodes = define_values_fixture
    env = environment_instances_fixture_no_selection
    for nnodes in defined_num_nodes:
        for nagents in defined_num_agents:
            td = env.reset_observe(num_agents=nagents, num_nodes=nnodes, device=defined_device, batch_size=defined_batch_size)
            while not td["done"].all():  
                td = env.sample_action(td, action_without_agent=True)
                td = env.sample_agent(td, agent_given_action=True)
                td = env.step_observe(td)
            env.check_solution_validity()
