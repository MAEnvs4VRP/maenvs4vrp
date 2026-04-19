import torch
from torch.distributions import Uniform
from tensordict import TensorDict

import os
from os import path
import pickle
from typing import Optional, Union, Callable, Dict
from maenvs4vrp.core.env_generator_builder import InstanceBuilder


GENERATED_INSTANCES_PATH = "hcvrp/data/generated"


class InstanceGenerator(InstanceBuilder):
    """
    Heterogeneous Capacitated Vehicle Routing Problem (HCVRP) instance generation class

    based on: https://github.com/ai4co/parco/blob/main/parco/envs/hcvrp/generator.py

    """

    @classmethod
    def get_list_of_benchmark_instances(cls):
        """
        Get list of generated files.

        Args:
            n/a.

        Returns:
            None.
        """

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
        generated = os.listdir(path.join(base_dir, GENERATED_INSTANCES_PATH))
        benchmark_instances = {}

        for folder in generated:
            val_path = path.join(GENERATED_INSTANCES_PATH, folder, "validation")
            test_path = path.join(GENERATED_INSTANCES_PATH, folder, "test")

            benchmark_instances[folder] = {
                "validation": [
                    val_path + "/" + s.split(".")[0]
                    for s in os.listdir(path.join(base_dir, val_path))
                ],
                "test": [
                    test_path + "/" + s.split(".")[0]
                    for s in os.listdir(path.join(base_dir, test_path))
                ],
            }
        return benchmark_instances

    def _check_distribution(self, name, dist, allow_none=False):
        if dist is None:
            if allow_none:
                return
            raise ValueError(f"{name} distribution cannot be None")

        if not callable(dist):
            raise TypeError(
                f"{name} distribution must be callable "
                f"(e.g. torch.distributions.Uniform)"
            )

    def __init__(
        self,
        instance_type: str = "validation",
        set_of_instances: set = None,
        device: Optional[str] = "cpu",
        batch_size: Optional[torch.Size] = None,
        seed: int = None,
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        depot_distribution: Union[int, float, str, type, Callable] = None,
        demand_distribution: Union[int, float, type, Callable] = Uniform,
        capacity_distribution: Union[int, float, type, Callable] = Uniform,
        speed_distribution: Union[int, float, type, Callable] = Uniform,
    ) -> None:
        """
        Instance generator for the Heterogeneous Capacitated Vehicle Routing Problem (HCVRP).

        Args:
            instance_type (str): Instance type. Can be "validation" or "test". Defaults to "validation".
            set_of_instances (set): Set of instances file names. Defaults to None.
            device (str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".
            batch_size (torch.Size, optional): Batch size. If not specified, defaults to 1.
            seed (int): Random number generator seed. Defaults to None.
            loc_distribution (Union[int, float, str, type, Callable]): Distribution for generating location coordinates. Defaults to torch.distributions.Uniform.
            depot_distribution (Union[int, float, str, type, Callable], optional): Distribution for generating depot locations. Defaults to None.
            demand_distribution (Union[int, float, type, Callable]): Distribution for generating customer demands. Defaults to torch.distributions.Uniform.
            capacity_distribution (Union[int, float, type, Callable]): Distribution for generating vehicle capacities. Defaults to torch.distributions.Uniform.
            speed_distribution (Union[int, float, type, Callable]): Distribution for generating vehicle speeds. Defaults to torch.distributions.Uniform.

        Returns:
            None.
        """

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

        self.num_nodes = 100
        self.num_agents_default = 3

        self._check_distribution("loc", loc_distribution)
        self._check_distribution("depot", depot_distribution, allow_none=True)
        self._check_distribution("demand", demand_distribution)
        self._check_distribution("capacity", capacity_distribution)
        self._check_distribution("speed", speed_distribution)

        def make_sampler(distribution):
            def sampler(low, high):
                low_t = torch.tensor(low, device=self.device, dtype=torch.float32)
                high_t = torch.tensor(high, device=self.device, dtype=torch.float32)

                if torch.any(high_t <= low_t):
                    raise ValueError(
                        f"Invalid bounds for distribution: low={low}, high={high}"
                    )

                return distribution(low_t, high_t)

            return sampler

        self.loc_sampler = make_sampler(loc_distribution)

        self.depot_sampler = (
            make_sampler(depot_distribution) if depot_distribution is not None else None
        )

        self.demand_sampler = make_sampler(demand_distribution)
        self.capacity_sampler = make_sampler(capacity_distribution)
        self.speed_sampler = make_sampler(speed_distribution)

        assert (
            instance_type in ["test", "validation"]
            or instance_type is None
            or instance_type == ""
        ), "Instance type must be 'test', 'validation', '' or None."
        # If None or empty, it loads both test and validation
        self.set_of_instances = set_of_instances

        if set_of_instances:
            self.instance_type = instance_type
            self.load_set_of_instances()

    def load_set_of_instances(self, set_of_instances: set = None):
        """
        Load every instance on set_of_instances set.

        Args:
            set_of_instances(set): Set of instances file names. Defaults to None.

        Returns:
            None.
        """

        if set_of_instances:
            self.set_of_instances = set_of_instances

        self.instances_data = dict()
        for instance_name in self.set_of_instances:
            instance = self.read_instance_data(instance_name)
            self.instances_data[instance_name] = instance

    def read_instance_data(self, instance_name: str) -> Dict:
        """
        Read instance data from file.

        Args:
            instance_name(str): instance file name.

        Returns:
            Dict: Instance data.
        """
        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
        generated_file = f"{base_dir}/{instance_name}.pkl"
        with open(generated_file, "rb") as fp:
            instance = pickle.load(fp)

        self.batch_size = instance["data"].batch_size
        instance["data"] = instance["data"].to(self.device)
        return instance

    def get_instance(self, instance_name: str, num_agents: int = None) -> Dict:
        """
        Get an instance with custom number of agents.

        Args:
            instance_name(str): Instance file name.
            num_agents(int): Number of agents. Defaults to None.

        Returns:
            Dict: Instance data.

        """

        instance = self.instances_data.get(instance_name)

        if num_agents is not None:
            assert num_agents > 0, "number of agents must be grater them 0!"
            instance["num_agents"] = num_agents

        return instance

    def random_generate_instance(
        self,
        num_agents: int = None,
        num_nodes: int = None,
        min_nodes: float = None,
        max_nodes: float = None,
        min_capacity: float = None,
        max_capacity: float = None,
        min_demand: int = None,
        max_demand: int = None,
        min_speed: float = None,
        max_speed: float = None,
        batch_size: Optional[torch.Size] = None,
        seed: int = None,
        device: Optional[str] = "cpu",
    ) -> Dict:
        """
        Generate a random instance.

        Args:
            num_agents(int): Number of heterogeneous agents/vehicles. Defaults to None.
            num_nodes(int): Total number of nodes including depot. Defaults to None.
            min_nodes(float): Minimum value for node coordinate generation. Defaults to None.
            max_nodes(float): Maximum value for node coordinate generation. Defaults to None.
            min_capacity(float): Minimum vehicle capacity. Defaults to None.
            max_capacity(float): Maximum vehicle capacity. Defaults to None.
            min_demand(int): Minimum customer demand. Defaults to None.
            max_demand(int): Maximum customer demand. Defaults to None.
            min_speed(float): Minimum vehicle speed. Defaults to None.
            max_speed(float): Maximum vehicle speed. Defaults to None.
            batch_size(Optional[torch.Size]): Batch size for instance generation. If None, uses the instance's
            default batch size. Defaults to None.
            seed(int): Random seed for reproducibility. If provided, sets the random seed before generation. Defaults to None.
            device(Optional[str]): Device for tensor operations ('cpu' or 'gpu'). Defaults to 'cpu'.

        Returns:
            TensorDict: Instance data.
        """

        if seed is not None:
            self._set_seed(seed)

        if num_agents is not None:
            assert num_agents > 0, "number of agents must be grater them 0!"
            self.num_agents_default = num_agents

        if num_nodes is not None:
            assert num_nodes > 0, "number of services must be grater them 0!"
            self.max_num_nodes = num_nodes

        if max_capacity is not None:
            assert max_capacity > 0, "Capacity must be greater than 0!"

        if max_speed is not None:
            assert max_speed > 0, "Speed must be greater than 0!"

        if batch_size is None:
            batch_size = self.batch_size
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)

        instance["num_agents"] = torch.full((*self.batch_size, 1), num_agents)

        # depot
        self.depot_idx = 0
        instance["depot_idx"] = torch.zeros(
            (*self.batch_size, 1), dtype=torch.int64, device=self.device
        )

        # coordinates
        loc_dist = self.loc_sampler(min_nodes, max_nodes)
        coords = loc_dist.sample((*self.batch_size, num_nodes, 2)).to(self.device)

        if self.depot_sampler is not None:
            depot_dist = self.depot_sampler(min_nodes, max_nodes)
            coords[:, 0:1, :] = depot_dist.sample((*self.batch_size, 1, 2)).to(
                self.device
            )

        instance["coords"] = coords

        # capacity
        cap_dist = self.capacity_sampler(0, max_capacity - min_capacity)
        capacity = cap_dist.sample((*self.batch_size, num_agents)).to(self.device)
        capacity = capacity + min_capacity
        instance["capacity"] = capacity

        # demands
        demand_dist = self.demand_sampler(min_demand - 1, max_demand - 1)
        demands = demand_dist.sample((*self.batch_size, num_nodes)).to(self.device)
        demands[:, self.depot_idx] = 0.0
        instance["demands"] = demands

        # speed (heterogeneous)
        speed_dist = self.speed_sampler(min_speed, max_speed)
        speed = speed_dist.sample((*self.batch_size, num_agents)).to(self.device)
        instance["speed"] = speed

        # depot mask
        is_depot = torch.zeros(
            (*self.batch_size, num_nodes), dtype=torch.bool, device=self.device
        )
        is_depot[:, self.depot_idx] = True
        instance["is_depot"] = is_depot

        instance_info = {
            "name": "random_instance",
            "num_nodes": num_nodes,
            "num_agents": num_agents,
            "data": instance,
        }
        return instance_info

    def augment_generate_instance(
        self,
        num_agents: int = None,
        num_nodes: int = None,
        min_nodes: float = None,
        max_nodes: float = None,
        min_demand: int = None,
        max_demand: int = None,
        min_capacity: float = None,
        max_capacity: float = None,
        min_speed: float = None,
        max_speed: float = None,
        batch_size: Optional[torch.Size] = None,
        n_augment: int = 2,
        seed: int = None,
        device: Optional[str] = "cpu",
    ) -> Dict:
        """
        Generate augmented instances.

        Args:
            num_agents(int): Number of heterogeneous agents/vehicles. Defaults to None.
            num_nodes(int): Total number of nodes including depot. Defaults to None.
            min_nodes(float): Minimum value for node coordinate generation. Defaults to None.
            max_nodes(float): Maximum value for node coordinate generation. Defaults to None.
            min_demand(int): Minimum customer demand. Defaults to None.
            max_demand(int): Maximum customer demand. Defaults to None.
            min_capacity(float): Minimum vehicle capacity. Defaults to None.
            max_capacity(float): Maximum vehicle capacity. Defaults to None.
            min_speed(float): Minimum vehicle speed. Defaults to None.
            max_speed(float): Maximum vehicle speed. Defaults to None.
            batch_size(Optional[torch.Size]): Final batch size for the augmented instance. Must be divisible by n_augment. Defaults to None.
            n_augment(int): Number of times to replicate the base instance. The base instance is generated with batch_size // n_augment samples. Defaults to 2.
            seed(int): Random seed for reproducibility. If provided, sets the random seed before generation. Defaults to None.
            device(Optional[str]): Device for tensor operations ('cpu' or 'gpu'). Defaults to 'cpu'.

        Returns:
            TensorDict: Instance data.
        """

        if seed is not None:
            self._set_seed(seed)

        if num_agents is not None:
            assert num_agents > 0, "number of agents must be grater them 0!"
            self.num_agents_default = num_agents

        if num_nodes is not None:
            assert num_nodes > 0, "number of services must be grater them 0!"
            self.max_num_nodes = num_nodes

        if max_capacity is not None:
            assert max_capacity > 0, "Capacity must be greater than 0!"

        if max_speed is not None:
            assert max_speed > 0, "Speed must be greater than 0!"

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        assert self.batch_size.numel() % n_augment == 0

        s_batch_size = self.batch_size.numel() // n_augment
        self.s_batch_size = torch.Size([s_batch_size])

        instance_info_s = self.random_generate_instance(
            num_agents=num_agents,
            num_nodes=num_nodes,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            min_demand=min_demand,
            max_demand=max_demand,
            min_capacity=min_capacity,
            max_capacity=max_capacity,
            min_speed=min_speed,
            max_speed=max_speed,
            batch_size=self.s_batch_size,
            seed=seed,
            device=device,
        )

        self.batch_size = torch.Size(batch_size)
        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)

        for key in instance_info_s["data"].keys():
            if len(instance_info_s["data"][key].shape) == 3:
                instance[key] = instance_info_s["data"][key].repeat(n_augment, 1, 1)
            elif len(instance_info_s["data"][key].shape) == 2:
                instance[key] = instance_info_s["data"][key].repeat(n_augment, 1)
            elif len(instance_info_s["data"][key].shape) == 1:
                instance[key] = instance_info_s["data"][key].repeat(n_augment)

        instance_info = {
            "name": "augmented_instance",
            "num_nodes": num_nodes,
            "num_agents": num_agents,
            "data": instance,
        }

        return instance_info

    def sample_name_from_set(self, seed: int = None) -> str:
        """
        Sample one instance from instance set.

        Args:
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            str: Instance name.
        """

        if seed is not None:
            self._set_seed(seed)
        assert (
            len(self.set_of_instances) > 0
        ), "set_of_instances has to have at least one instance!"

        return list(self.set_of_instances)[
            torch.randint(0, len(self.set_of_instances), (1,)).item()
        ]

    def sample_instance(
        self,
        num_agents: int = 3,
        num_nodes: int = 40,
        min_nodes: float = 0.0,
        max_nodes: float = 1.0,
        min_demand: int = 1,
        max_demand: int = 10,
        min_capacity: float = 20,
        max_capacity: float = 41,
        min_speed: float = 0.5,
        max_speed: float = 1.0,
        instance_name: str = None,
        sample_type: str = "random",
        batch_size: Optional[torch.Size] = None,
        n_augment: Optional[int] = None,
        seed: int = None,
        device: Optional[str] = "cpu",
    ) -> Dict:
        """
        Sample one instance from instance space.

        Args:
            num_agents(int): Number of heterogeneous agents/vehicles. Defaults to 3.
            num_nodes(int): Total number of nodes including depot. Defaults to 40.
            min_nodes(float): Minimum value for node coordinate generation. Defaults to 0.0.
            max_nodes(float): Maximum value for node coordinate generation. Defaults to 1.0.
            min_demand(int): Minimum customer demand. Defaults to 1.
            max_demand(int): Maximum customer demand. Defaults to 10.
            min_capacity(float): Minimum vehicle capacity. Defaults to 20.
            max_capacity(float): Maximum vehicle capacity. Defaults to 41.
            min_speed(float): Minimum vehicle speed. Defaults to 0.5.
            max_speed(float): Maximum vehicle speed. Defaults to 1.0.
            instance_name(str): Name of the instance to load. Used when sample_type is 'saved'.
            If None and a set of instances exists, a random instance is selected. Defaults to None.
            sample_type(str): Strategy for instance generation/sampling. Can be:
                - 'random': Generate a random instance using random_generate_instance().
                - 'augment': Generate augmented instances using augment_generate_instance().
                - 'saved': Load a pre-saved instance from the set_of_instances.
                Defaults to 'random'.
            batch_size(Optional[torch.Size]): Batch size for instance generation. If None, uses the instance's
                default batch size. Defaults to None.
            n_augment(Optional[int]): Number of augmentation replications. Used only when sample_type is 'augment'. Defaults to None.
            seed(int): Random seed for reproducibility. If provided, sets the random seed before sampling. Defaults to None.
            device(Optional[str]): Device for tensor operations ('cpu' or 'gpu'). Defaults to 'cpu'.

        Returns:
            Dict: Instance data.
        """

        if seed is not None:
            self._set_seed(seed)

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

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
            self.num_agents_default = 3
        else:
            self.num_agents_default = num_agents

        if num_nodes is None:
            self.num_nodes = 40
        else:
            self.num_nodes = num_nodes

        if min_nodes is None:
            self.min_nodes = 0.0
        else:
            self.min_nodes = min_nodes

        if max_nodes is None:
            self.max_nodes = 1.0
        else:
            self.max_nodes = max_nodes

        if min_demand is None:
            self.min_demand = 1
        else:
            self.min_demand = min_demand

        if max_demand is None:
            self.max_demand = 10
        else:
            self.max_demand = max_demand

        if min_capacity is None:
            self.min_capacity = 20
        else:
            self.min_capacity = min_capacity

        if max_capacity is None:
            self.max_capacity = 41
        else:
            self.max_capacity = max_capacity

        if min_speed is None:
            self.min_speed = 0.5
        else:
            self.min_speed = min_speed

        if max_speed is None:
            self.max_speed = 1.0
        else:
            self.max_speed = max_speed

        if sample_type == "random":
            instance_info = self.random_generate_instance(
                num_agents=self.num_agents_default,
                num_nodes=self.num_nodes,
                min_nodes=self.min_nodes,
                max_nodes=self.max_nodes,
                min_demand=self.min_demand,
                max_demand=self.max_demand,
                min_capacity=self.min_capacity,
                max_capacity=self.max_capacity,
                min_speed=self.min_speed,
                max_speed=self.max_speed,
                batch_size=self.batch_size,
                seed=self.seed,
                device=self.device,
            )

        if sample_type == "augment":
            instance_info = self.augment_generate_instance(
                num_agents=self.num_agents_default,
                num_nodes=self.num_nodes,
                min_nodes=self.min_nodes,
                max_nodes=self.max_nodes,
                min_demand=self.min_demand,
                max_demand=self.max_demand,
                min_capacity=self.min_capacity,
                max_capacity=self.max_capacity,
                min_speed=self.min_speed,
                max_speed=self.max_speed,
                batch_size=self.batch_size,
                n_augment=n_augment,
                seed=self.seed,
                device=self.device,
            )

        if sample_type == "saved":
            instance_info = self.get_instance(instance_name, num_agents=num_agents)

        return instance_info


if __name__ == "__main__":
    number_instances = 64
    print("starting valid/test sets generation")

    # valid/test sets generation
    for num_nodes, n_agent in [(101, 25), (51, 25)]:
        generator = InstanceGenerator(batch_size=32, seed=0)
        for k in range(number_instances):
            instance = generator.sample_instance(
                num_agents=n_agent, num_nodes=num_nodes
            )
            name = f"generated_val_servs_{num_nodes - 1}_agents_{n_agent}_{k}"
            instance["name"] = name
            if not os.path.exists(
                f"data/generated/servs_{num_nodes - 1}_agents_{n_agent}/validation"
            ):
                os.makedirs(
                    f"data/generated/servs_{num_nodes - 1}_agents_{n_agent}/validation"
                )
            with open(
                f"data/generated/servs_{num_nodes - 1}_agents_{n_agent}/validation/"
                + name
                + ".pkl",
                "wb",
            ) as fp:
                pickle.dump(instance, fp, protocol=pickle.HIGHEST_PROTOCOL)

            instance = generator.sample_instance(
                num_agents=n_agent, num_nodes=num_nodes
            )
            name = f"generated_test_servs_{num_nodes - 1}_agents_{n_agent}_{k}"
            instance["name"] = name
            if not os.path.exists(
                f"data/generated/servs_{num_nodes - 1}_agents_{n_agent}/test"
            ):
                os.makedirs(
                    f"data/generated/servs_{num_nodes - 1}_agents_{n_agent}/test"
                )
            with open(
                f"data/generated/servs_{num_nodes - 1}_agents_{n_agent}/test/"
                + name
                + ".pkl",
                "wb",
            ) as fp:
                pickle.dump(instance, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print("done")
