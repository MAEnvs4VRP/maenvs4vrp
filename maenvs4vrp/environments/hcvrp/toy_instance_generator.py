import torch
import os
from tensordict import TensorDict
from typing import Optional, Dict
from maenvs4vrp.core.env_generator_builder import InstanceBuilder


class ToyInstanceGenerator(InstanceBuilder):
    """
    HCVRP toy instance generation class.
    """

    def __init__(
        self,
        instance_type: str = "validation",
        set_of_instances: set = None,
        device: Optional[str] = "cpu",
        batch_size: Optional[torch.Size] = None,
        seed: int = None
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
            batch_size = (
                [batch_size] if isinstance(batch_size, int) else batch_size
            )

        self.batch_size = torch.Size(batch_size)


        self.max_num_agents = 4
        self.max_num_nodes = 13

        assert instance_type in ["validation", "test"]
        self.set_of_instances = set_of_instances

        if set_of_instances:
            self.instance_type = instance_type
            self.load_set_of_instances()

    def random_generate_instance(
        self,
        num_agents: int = 4,
        num_nodes: int = 13,
        batch_size: Optional[int] = None,
        seed: int = None,
    ) -> Dict:
        """
        Generate random toy instance.

        Args:
            num_agents(int): Total number of agents. Defaults to 4.
            num_nodes(int): Total number of nodes. Defaults to 13.
            batch_size(int): Batch size. Defaults to 1.
            seed(int, optional): Random number generator seed. Defaults to None.

        Returns:
            TensorDict: Instance data.
        """

        if seed is not None:
            self._set_seed(seed)

        
        if num_agents is not None:
            assert num_agents>0, f"number of agents must be grater them 0!"
            self.max_num_agents = num_agents
        if num_nodes is not None:
            assert num_nodes>0, f"number of services must be grater them 0!"
            self.max_num_nodes = num_nodes


        if batch_size is not None:
            batch_size = (
                [batch_size] if isinstance(batch_size, int) else batch_size
            )
            self.batch_size = torch.Size(batch_size)


        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)

        # Depot index
        self.depot_idx = 0
        instance["depot_idx"] = torch.zeros(
            (*self.batch_size, 1), dtype=torch.int64, device=self.device
        )

        # Coords
        coords = torch.tensor(
            [
                [
                    [0, 0],   # depot
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
            dtype=torch.float32,
            device=self.device
        )
        instance["coords"] = coords

        # Demands
        demands = torch.tensor(
            [[0., 5., 6., 4., 7., 3., 4., 6., 5., 3., 6., 5., 4.]],
            dtype=torch.float32,
            device=self.device
        )
        instance["demands"] = demands

        # Capacities (heterogeneous)
        capacity = torch.tensor(
            [[10., 15., 20., 25.]],
            dtype=torch.float32,
            device=self.device
        )
        instance["capacity"] = capacity

        # Speeds
        speed = torch.tensor(
            [[1.0, 0.8, 1.2, 1.1]],
            dtype=torch.float32,
            device=self.device
        )
        instance["speed"] = speed

        # depot mask
        is_depot = torch.zeros(
            (*self.batch_size, num_nodes), dtype=torch.bool, device=self.device
        )
        is_depot[:, self.depot_idx] = True
        instance["is_depot"] = is_depot

        # explicit num_agents field
        instance["num_agents"] = torch.full(
            (*self.batch_size, 1), num_agents, dtype=torch.int64, device=self.device
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
        num_agents: int = 4,
        num_nodes: int = 13,
        instance_name: str = None,
        sample_type: str = "random",
        batch_size: Optional[int] = None,
        seed: int = None,
    ) -> Dict:
        """
        Sample one instance from instance space.

        Args:
            num_agents(int): Total number of agents. Defaults to None.
            num_nodes(int): Total number of nodes. Defaults to None.
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

        if instance_name==None and random_sample==False:
            instance_name = self.sample_name_from_set(seed=seed)
        elif instance_name==None and random_sample==True:
            instance_name = 'random_instance'
        else:
            instance_name = instance_name

        if num_agents is None:
            num_agents = 4
        else:
            num_agents = num_agents
        if num_nodes is None:
            num_nodes = 137
        else:
            num_nodes = num_nodes

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)
        

        if sample_type == "random":
            return self.random_generate_instance(
                num_agents=num_agents,
                num_nodes=num_nodes,
                batch_size=batch_size,
                seed=seed,
            )


        if sample_type == "saved":
            if instance_name is None:
                instance_name = self.sample_name_from_set(seed=seed)
            return self.get_instance(instance_name, num_agents=num_agents)

        raise ValueError(f"Unknown sample_type: {sample_type}")


if __name__ == "__main__":
        
    number_instances = 128
    print('starting valid/test sets generation')

    if not os.path.exists('data/generated/test'):
        os.makedirs('data/generated/test')
    if not os.path.exists('data/generated/validation'):
        os.makedirs('data/generated/validation')

    
    print('done')
