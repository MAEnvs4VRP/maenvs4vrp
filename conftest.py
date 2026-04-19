def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Which device use for training. It can be 'cpu' or 'gpu'.",
    )

    parser.addoption(
        "--batch",
        "--batch_size",
        action="store",
        default=None,
        help="Batch size to run the tests",
    )

    parser.addoption(
        "--agents",
        "--num_agents",
        action="append",
        default=None,
        help="Number of agents. It can be one or more integers.",
    )

    parser.addoption(
        "--nodes",
        "--num_nodes",
        action="append",
        default=None,
        help="Number of nodes. It can be one or more integers.",
    )


import pytest


@pytest.fixture(scope="session")
def device(request):
    return request.config.getoption("--device")


@pytest.fixture(scope="session")
def batch(request):
    return request.config.getoption("--batch")


@pytest.fixture(scope="session")
def num_agents(request):
    return request.config.getoption("--num_agents")


@pytest.fixture(scope="session")
def num_nodes(request):
    return request.config.getoption("--num_nodes")


def pytest_configure(config):
    config.option.log_cli = True
    config.option.log_cli_level = "WARNING"
