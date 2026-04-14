import pytest
import importlib
from maenvs4vrp.utils.utils import data_equivalence

ENVIRONMENT_LIST = ['dvrptw', 'dsvrptw', 'cvrptw', 'toptw', 'cvrpstw', 'sdvrptw', 
                    'pcvrptw', 'pdptw', 'mdvrptw', 'mtvrp', 'gmtvrp', 'mtdvrp', 'gmtdvrp',
                    'cvrp', 'hcvrp', 'top']


@pytest.fixture(params=ENVIRONMENT_LIST)
def instances_generator_fixture(request):
    generator_module_name = f'maenvs4vrp.environments.{request.param}.instances_generator'
    generator = importlib.import_module(generator_module_name).InstanceGenerator()
    return generator


def test_different_seed_instances_generator(instances_generator_fixture):
    instance1 = instances_generator_fixture.sample_instance(num_agents=50, num_nodes=101, seed=1)
    instance2 = instances_generator_fixture.sample_instance(num_agents=50, num_nodes=101, seed=5)
    assert not data_equivalence(instance1, instance2)


def test_same_seed_instances_generator(instances_generator_fixture):
    instance1 = instances_generator_fixture.sample_instance(num_agents=50, num_nodes=101, seed=1)
    instance2 = instances_generator_fixture.sample_instance(num_agents=50, num_nodes=101, seed=1)
    assert data_equivalence(instance1, instance2)



