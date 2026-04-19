import torch
from tensordict import TensorDict

import os

from typing import Dict, Optional
from maenvs4vrp.core.env_generator_builder import InstanceBuilder

GENERATED_INSTANCES_PATH = "dsvrptw/data/generated"


class ToyInstanceGenerator(InstanceBuilder):
    """
    DSVRPTW toy instance generation class.
    """

    def __init__(
        self,
        instance_type: str = "validation",
        set_of_instances: set = None,
        device: Optional[str] = "cpu",
        batch_size: Optional[torch.Size] = None,
        seed: int = None,
    ) -> None:
        """
        Constructor. Toy instance generator for testing.

        Args:
            instance_type(str):  instance type. Can be "validation" or "test". Defaults to "validation".
            set_of_instances(set): Set of instances file names. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".
            batch_size(torch.Size, optional): Batch size. If not specified, defaults to 1. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.
        """

        # seed the generation process
        if seed is None:
            self._set_seed(self.DEFAULT_SEED)
        else:
            self._set_seed(seed)

        self.device = device
        if batch_size is None:
            batch_size = [1]
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.batch_size = torch.Size(batch_size)

        self.max_num_agents = 4
        self.max_num_nodes = 13

        assert instance_type in ["validation", "test"], "instance unknown type"
        self.set_of_instances = set_of_instances
        if set_of_instances:
            self.instance_type = instance_type
            self.load_set_of_instances()

    def random_generate_instance(
        self,
        num_agents: int = 4,
        num_nodes: int = 13,
        capacity: int = 10,
        service_times: int = 0.2,
        batch_size: int = 1,
        seed: int = None,
    ) -> TensorDict:
        """
        Generate random toy instance.

        Args:
            num_agents(int): Total number of agents. Defaults to 4.
            num_nodes(int): Total number of nodes. Defaults to 13.
            capacity(int): Total capacity for each agent. Defaults to 10.
            service_times(int): Service times in the nodes. Defaults to 0.2.
            batch_size(int): Batch size. Defaults to 1.
            seed(int, optional): Random number generator seed. Defaults to None.

        Returns:
            TensorDict: Instance data.
        """
        if seed is not None:
            self._set_seed(seed)

        if num_agents is not None:
            assert num_agents > 0, "number of agents must be grater them 0!"
            self.max_num_agents = num_agents
        if num_nodes is not None:
            assert num_nodes > 0, "number of services must be grater them 0!"
            self.max_num_nodes = num_nodes
        if service_times is not None:
            self.service_times = service_times
        if capacity is not None:
            assert capacity > 0, "agent capacity must be grater them 0!"
            self.capacity = capacity

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)

        self.depot_idx = 0
        instance["depot_idx"] = self.depot_idx * torch.ones(
            (*self.batch_size, 1), dtype=torch.int64, device=self.device
        )

        coords = torch.tensor(
            [
                [
                    [0, 0],
                    [1, 2],
                    [2, 3],
                    [3, 2],
                    [-1, 2],
                    [-2, 3],
                    [-3, 2],
                    [-1, -2],
                    [-2, -3],
                    [-3, -2],
                    [1, -2],
                    [2, -3],
                    [3, -2],
                ]
            ],
            device=self.device,
        )
        instance["coords"] = coords

        service_times = self.service_times * torch.ones(
            (*self.batch_size, num_nodes), dtype=torch.float, device=self.device
        )
        service_times[:, self.depot_idx] = 0
        instance["service_time"] = service_times

        time_windows = torch.tensor(
            [
                [
                    [0, 24],
                    [8, 11],
                    [10, 12],
                    [9, 14],
                    [12, 16],
                    [8, 10],
                    [12, 19],
                    [3, 6],
                    [7, 9],
                    [15, 18],
                    [3, 6],
                    [7, 18],
                    [16, 19],
                ]
            ],
            device=self.device,
        )

        instance["tw_low"] = time_windows[:, :, 0].clone()
        instance["tw_high"] = time_windows[:, :, 1].clone()

        instance["is_depot"] = torch.zeros(
            (*self.batch_size, num_nodes), dtype=torch.bool, device=self.device
        )
        instance["is_depot"][:, self.depot_idx] = True

        instance["start_time"] = (
            time_windows[:, :, 0]
            .gather(
                1,
                torch.zeros(
                    (*self.batch_size, 1), dtype=torch.int64, device=self.device
                ),
            )
            .squeeze(-1)
        )
        instance["end_time"] = (
            time_windows[:, :, 1]
            .gather(
                1,
                torch.zeros(
                    (*self.batch_size, 1), dtype=torch.int64, device=self.device
                ),
            )
            .squeeze(-1)
        )

        demands = torch.tensor(
            [[0.0, 5.0, 6.0, 4.0, 7.0, 3.0, 4.0, 6.0, 5.0, 3.0, 6.0, 5.0, 4.0]],
            device=self.device,
        )

        instance["demands"] = demands
        instance["capacity"] = self.capacity * torch.ones(
            (*self.batch_size, 1), dtype=torch.float, device=self.device
        )
        instance["appear_time"] = torch.tensor(
            [[0.0, 0.0, 4.0, 8.0, 1.0, 3.0, 8.0, 0.0, 4.0, 8.0, 0.0, 4.0, 10.0]],
            device=self.device,
        )

        instance_info = {
            "name": "toy_instance",
            "num_nodes": self.max_num_nodes,
            "num_agents": self.max_num_agents,
            "data": instance,
        }
        return instance_info

    def sample_instance(
        self,
        num_agents=None,
        num_nodes=None,
        service_times=0.2,
        capacity=20,
        instance_name: str = None,
        sample_type: str = "random",
        batch_size: Optional[torch.Size] = None,
        n_augment: Optional[int] = None,
        seed: int = None,
    ) -> Dict:
        """
        Sample one instance from instance space.

        Args:
            num_agents(int): Total number of agents. Defaults to None.
            num_nodes(int): Total number of nodes. Defaults to None.
            service_times(float): Service times in the nodes. Defaults to 0.2.
            capacity(int): Capacity of the agents. Defaults to 10.
            instance_name(str): Instance name. Defaults to None.
            sample_type(str): Sample type. It can be "random" or something else for "first n". Defaults to "random".
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Data augmentation. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            Dict: Instance data.
        """
        if seed is not None:
            self._set_seed(seed)

        if self.set_of_instances is None:
            random_sample = True
        else:
            random_sample = False

        if instance_name == None and random_sample == False:
            instance_name = self.sample_name_from_set(seed=seed)
        elif instance_name == None and random_sample == True:
            instance_name = "random_instance"
        else:
            instance_name = instance_name

        if num_agents is None:
            num_agents = 4
        if num_nodes is None:
            num_nodes = 13
        if service_times is None:
            service_times = 0.2
        if capacity is None:
            capacity = 10

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        if sample_type == "random":
            instance_info = self.random_generate_instance(
                num_agents=num_agents,
                num_nodes=num_nodes,
                capacity=capacity,
                service_times=service_times,
                batch_size=batch_size,
                seed=seed,
            )

        return instance_info


if __name__ == "__main__":
    number_instances = 128
    print("starting valid/test sets generation")

    if not os.path.exists("data/generated/test"):
        os.makedirs("data/generated/test")
    if not os.path.exists("data/generated/validation"):
        os.makedirs("data/generated/validation")

    print("done")
