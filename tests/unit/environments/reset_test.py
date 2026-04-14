import pytest
import importlib

ENVIRONMENT_LIST = ['cvrp', 'hcvrp', 'top', 'cvrpstw', 'cvrptw', 'toptw', 'dvrptw', 'dsvrptw', 'mdvrptw', 'pdptw', 'sdvrptw', 'pcvrptw']


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


@pytest.fixture(params=ENVIRONMENT_LIST)
def environment_instances_all_observations_fixture(request):
    env_agent_selector_module_name = f'maenvs4vrp.environments.{request.param}.env_agent_selector'
    env_agent_selector = importlib.import_module(env_agent_selector_module_name).RandomSelector()

    observations_module_name = f'maenvs4vrp.environments.{request.param}.observations'
    Observations = importlib.import_module(observations_module_name).Observations
    all_possible_features = {'nodes_static': dict([(c, {'feat': c, 'norm': None}) for c in Observations.POSSIBLE_NODES_STATIC_FEATURES]),
                             'nodes_dynamic': Observations.POSSIBLE_NODES_DYNAMIC_FEATURES,
                             'agent': Observations.POSSIBLE_AGENT_FEATURES,
                             'all_agents': Observations.POSSIBLE_ALL_AGENTS_FEATURES,
                             'other_agents': Observations.POSSIBLE_OTHER_AGENTS_FEATURES,
                             'global': Observations.POSSIBLE_GLOBAL_FEATURES}
    observations = Observations(all_possible_features)

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


# reset tests
def test_instance_env_reset_gives_no_error(environment_instances_fixture):
    env = environment_instances_fixture
    td = env.reset()

# reset observe tests
def test_instance_env_reset_observe_gives_no_error(environment_instances_fixture):
    env = environment_instances_fixture
    td = env.reset_observe()

# reset select tests
def test_instance_env_reset_agent_select_gives_no_error(environment_instances_fixture):
    env = environment_instances_fixture
    td = env.reset_agent_select()

# reset select observe tests
def test_instance_env_reset_agent_select_observe_gives_no_error(environment_instances_fixture):
    env = environment_instances_fixture
    td = env.reset_agent_select_observe()
    

# agent selection through rules tests
def test_instance_env_agent_iterator_gives_no_error(environment_instances_fixture):
    env = environment_instances_fixture
    td = env.reset_agent_select_observe()
    while not td["done"].all():  
        td = env.sample_action(td)
        td = env.step_agent_select_observe(td)

def test_instance_env_agent_smallesttime_iterator_gives_no_error(environment_instances_fixture_st):
    env = environment_instances_fixture_st
    td = env.reset_agent_select_observe()
    while not td["done"].all():  
        td = env.sample_action(td)
        td = env.step_agent_select_observe(td)

def test_instance_env_agent_rand_iterator_gives_no_error(environment_instances_fixture_rand):
    env = environment_instances_fixture_rand
    td = env.reset_agent_select_observe()
    while not td["done"].all():  
        td = env.sample_action(td)
        td = env.step_agent_select_observe(td)

# sequential agent-node selection tests
def test_instance_env_agent_node_selection_gives_no_error(environment_instances_fixture_no_selection):
    env = environment_instances_fixture_no_selection
    td = env.reset()
    while not td["done"].all():  
        td = env.sample_agent(td)
        td = env.sample_action(td)
        td = env.step_observe(td)

# simultaneous agent-node selection tests
def test_instance_env_joint_selection_gives_no_error(environment_instances_fixture_no_selection):
    env = environment_instances_fixture_no_selection
    td = env.reset_observe()
    while not td["done"].all():  
        td = env.sample_joint(td)
        td = env.step_observe(td)

# sequential node-agent selection tests
def test_instance_env_node_agent_selection_gives_no_error(environment_instances_fixture_no_selection):
    env = environment_instances_fixture_no_selection
    td = env.reset_observe()
    while not td["done"].all():  
        td = env.sample_action(td, action_without_agent=True)
        td = env.sample_agent(td, agent_given_action=True)
        td = env.step_observe(td)