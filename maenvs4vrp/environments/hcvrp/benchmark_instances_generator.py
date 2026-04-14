import os
from os import path
import shutil
import logging
import torch
import numpy as np
from tensordict import TensorDict
from typing import Optional, Dict, Set

from huggingface_hub import snapshot_download
from maenvs4vrp.core.env_generator_builder import InstanceBuilder

log = logging.getLogger(__name__)

BENCHMARK_INSTANCES_PATH = "hcvrp/data/benchmark"

HF_REPO_ID = "ai4co/parco"


class BenchmarkInstanceGenerator(InstanceBuilder):

    """
    HCVRP Benchmark Instance Generator class.
    """

    @classmethod
    def get_list_of_benchmark_instances(cls):
        """
        Get list of possible instances from benchmark files.

        Args:
            n/a.

        Returns:
            None.
        """

        cls.download_and_copy_instances()

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
        full_dir = path.join(base_dir, BENCHMARK_INSTANCES_PATH)

        files = [f for f in os.listdir(full_dir) if f.endswith(".npz")]

        return {
            "instances": [
                f"{BENCHMARK_INSTANCES_PATH}/{fname.split('.')[0]}"
                for fname in files
            ]
        }

    @classmethod
    def download_and_copy_instances(cls):
        """
        Download benchmark instances from HuggingFace if they are not locally present.

        Args:
            n/a.

        Returns:
            None.
        """

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
        target_dir = path.join(base_dir, BENCHMARK_INSTANCES_PATH)


        if os.path.isdir(target_dir):
            return


        local_snapshot = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )


        os.makedirs(target_dir, exist_ok=True)

        for root, _, files in os.walk(local_snapshot):
            if "data{}hcvrp".format(os.sep) not in root:
                continue

            for fname in files:
                if not fname.endswith(".npz"):
                    continue

                src = path.join(root, fname)
                dst = path.join(target_dir, fname)

                if path.exists(dst):
                    log.info(f"Ignorando duplicado: {fname}")
                    continue

                shutil.copy(src, dst)
                log.info(f"Copiado: {fname}")

        log.warning("Download concluído.")


    def __init__(
        self,
        set_of_instances: Set[str] = None,
        device: str = "cpu",
        batch_size: int = 1,
        seed: int = None,
    ):
        """
        Initialize the BenchmarkInstanceGenerator.

        Args:
            set_of_instances (Set[str], optional): Set of instance identifiers to load
                (for example, 'hcvrp/data/benchmark/instance_name'). If provided,
                `load_set_of_instances()` will be called. Defaults to None.
            device (str, optional): PyTorch device string ('cpu' or 'cuda'). Defaults to 'cpu'.
            batch_size (int, optional): Number of examples per batch used when creating TensorDicts. Defaults to 1.
            seed (int, optional): Random seed for reproducibility. If None, `DEFAULT_SEED` will be used.

        """

        # If instances are not on local machine, they'll be downloaded from RouteFinder's HuggingFace
        self.download_and_copy_instances()

        if seed is None:
            self._set_seed(self.DEFAULT_SEED)
        else:
            self._set_seed(seed)

        self.device = device
        self.batch_size = torch.Size([batch_size])

        self.set_of_instances = set_of_instances

        if set_of_instances:
            self.load_set_of_instances()

   
    def load_set_of_instances(self, set_of_instances=None):
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
            instance = self.read_parse_instance_data(instance_name)
            self.instances_data[instance_name] = instance

    def read_parse_instance_data(self, instance_name: str) -> Dict:
        """
        Read instance data from file. Benchmark's instance keys are translated into our keys.

        Args:
            instance_name(str): Instance path.

        Returns: 
            Dict: Instance data.
        """

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
        file_path = f"{base_dir}/{instance_name}.npz"

        loaded = np.load(file_path)

        depot = loaded["depot"]
        locs = loaded["locs"]
        demand = loaded["demand"] 
        capacity = loaded["capacity"]
        speed = loaded["speed"]

        num_instances = locs.shape[0]
        num_nodes = locs.shape[1]
        num_agents = capacity.shape[1] 

        locs_all = locs.copy()
        locs_all[:, 0, :] = depot

        batch = num_instances
        data = TensorDict({}, batch_size=[batch], device=self.device)

        data["coords"] = torch.from_numpy(locs_all).float().to(self.device)
        data["demand"] = torch.from_numpy(demand).float().to(self.device)     
        data["capacity"] = torch.from_numpy(capacity).float().to(self.device) 
        data["speed"] = torch.from_numpy(speed).float().to(self.device)       

        # Depot mask
        is_depot = torch.zeros((batch, num_nodes), dtype=torch.bool, device=self.device)
        is_depot[:, 0] = True
        data["is_depot"] = is_depot

        return {
            "name": instance_name.split("/")[-1],
            "num_nodes": num_nodes,
            "num_agents": num_agents,
            "data": data,
        }
    
    def get_instance(self, instance_name: str) -> Dict:
        """
        Get an instance with custom number of agents.

        Args:
            instance_name(str): Instance file name.
            num_agents(int): Number of agents. Defaults to None.

        Returns:
            Dict: Instance data.

        """

        if not hasattr(self, "instances_data"):
            raise RuntimeError("No instances loaded. Call load_set_of_instances() first.")

        if instance_name not in self.instances_data:
            raise KeyError(f"Instance '{instance_name}' not found.")

        return self.instances_data[instance_name]



    def random_sample_instance(
        self,
        instance_name: str = None,
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
        seed: int = None,
        device: Optional[str] = "cpu", 
    ) -> Dict:
        """
        Generate a random HCVRP instance with specified parameters.

        Args:
            instance_name (str, optional): Instance name identifier. Defaults to None.
            num_agents (int, optional): Number of vehicles/agents. Defaults to None.
            num_nodes (int, optional): Total number of nodes (including depot). Defaults to None.
            min_nodes (float, optional): Minimum coordinate value for node locations. Defaults to None.
            max_nodes (float, optional): Maximum coordinate value for node locations. Defaults to None.
            min_demand (int, optional): Minimum customer demand. Defaults to None.
            max_demand (int, optional): Maximum customer demand. Defaults to None.
            min_capacity (float, optional): Minimum vehicle capacity. Defaults to None.
            max_capacity (float, optional): Maximum vehicle capacity. Defaults to None.
            min_speed (float, optional): Minimum vehicle speed. Defaults to None.
            max_speed (float, optional): Maximum vehicle speed. Defaults to None.
            batch_size (torch.Size, optional): Batch size for the instance. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            device (str, optional): Computation device ('cpu' or 'cuda'). Defaults to 'cpu'.

        Returns:
            Dict: Dictionary containing:
                - 'name' (str): Instance name.
                - 'num_nodes' (int): Number of nodes in the instance.
                - 'num_agents' (int): Number of vehicles/agents.
                - 'data' (TensorDict): TensorDict containing:
                    - 'coords': Node coordinates with depot at index 0.
                    - 'demand': Demand at each node (0 at depot).
                    - 'capacity': Capacity for each vehicle.
                    - 'speed': Speed for each vehicle.
                    - 'is_depot': Boolean mask indicating depot nodes.
        """

        if seed is not None:
            self._set_seed(seed)

        if num_agents is None:
            self.num_agents = 3
        else:
            self.num_agents = num_agents

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
            self.min_capacity = 20.0
        else:
            self.min_capacity = min_capacity

        if max_capacity is None:
            self.max_capacity = 50.0
        else:
            self.max_capacity = max_capacity

        if min_speed is None:
            self.min_speed = 0.5
        else:
            self.min_speed = min_speed

        if max_speed is None:
            self.max_speed = 1.5
        else:
            self.max_speed = max_speed

        batch = self.batch_size[0]
        data = TensorDict({}, batch_size=self.batch_size, device=self.device)

        # Depot
        depot = torch.rand((batch, 1, 2)) * (max_nodes - min_nodes) + min_nodes

        # Coords
        coords = torch.rand((batch, num_nodes, 2)) * (max_nodes - min_nodes) + min_nodes
        coords[:, 0:1, :] = depot

        data["coords"] = coords

        # Demands
        demand = torch.randint(min_demand, max_demand, (batch, num_nodes)).float()
        demand[:, 0] = 0.0
        data["demand"] = demand

        # Capacity
        capacity = torch.rand((batch, num_agents)) * (max_capacity - min_capacity) + min_capacity
        data["capacity"] = capacity

        # Speed
        speed = torch.rand((batch, num_agents)) * (max_speed - min_speed) + min_speed
        data["speed"] = speed

        # Depot mask
        is_depot = torch.zeros((batch, num_nodes), dtype=torch.bool)
        is_depot[:, 0] = True
        data["is_depot"] = is_depot

        instance = {
            "name": "random_hcvrp_instance",
            "num_nodes": num_nodes,
            "num_agents": num_agents,
            "data": data.to(self.device),
        }

        return instance

    def sample_name_from_set(self, seed=None):
        """
        Sample one instance from instance set.

        Args:
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            str: Instance sample name.
        """
        if seed is not None:
            self._set_seed(seed)
        inst = list(self.set_of_instances)
        return inst[torch.randint(0, len(inst), (1,)).item()]

    def sample_instance(
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
                        instance_name: str = None,
                        sample_type: str = "random",
                        batch_size: Optional[torch.Size] = None,
                        seed: int = None,
                        n_augment: Optional[int] = None,
                        device: Optional[str] = "cpu",
                    ) -> Dict:
        """
        Sample or generate an HCVRP instance.

        Depending on `sample_type`, this method either generates a new random
        instance (delegating to `random_sample_instance`) or returns a saved
        instance previously loaded into the generator's `instances_data`.

        Args:
            num_agents (int, optional): Number of vehicles/agents. If None,
                a default value is used.
            num_nodes (int, optional): Number of nodes including the depot.
                If None, a default value is used.
            min_nodes (float, optional): Minimum coordinate value for locations.
            max_nodes (float, optional): Maximum coordinate value for locations.
            min_demand (int, optional): Minimum customer demand (inclusive).
            max_demand (int, optional): Maximum customer demand (exclusive).
            min_capacity (float, optional): Minimum vehicle capacity.
            max_capacity (float, optional): Maximum vehicle capacity.
            min_speed (float, optional): Minimum vehicle speed.
            max_speed (float, optional): Maximum vehicle speed.
            instance_name (str, optional): Name of a saved instance to load.
                If `None`, one is sampled from the generator's set.
            sample_type (str, optional): Either `'random'` to synthesize a new
                instance or `'saved'` to return a stored instance. Defaults to
                `'random'`.
            batch_size (int or torch.Size, optional): If provided, overrides
                the generator's batch size for the produced TensorDicts.
            seed (int, optional): Random seed for reproducibility.
            n_augment (int, optional): Number of augmentations to apply
                (currently unused by this method).
            device (str, optional): Device string for tensors (e.g. `'cpu'` or
                `'cuda'`).

        Returns:
            Dict: A dictionary containing:
                - `'name'` (str): Instance identifier.
                - `'num_nodes'` (int): Number of nodes in the instance.
                - `'num_agents'` (int): Number of vehicles/agents.
                - `'data'` (TensorDict): Batched TensorDict with keys
                  `'coords'`, `'demand'`, `'capacity'`, `'speed'`, and
                  `'is_depot'`.

        Raises:
            ValueError: If `sample_type` is not `'random'` or `'saved'`.
        """

        if seed is not None:
            self._set_seed(seed)

        if instance_name is None:
            instance_name = self.sample_name_from_set(seed=seed)
        else:
            instance_name = instance_name


        if num_agents is None:
            self.num_agents = 3
        else:
            self.num_agents = num_agents

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
            self.min_capacity = 20.0
        else:
            self.min_capacity = min_capacity

        if max_capacity is None:
            self.max_capacity = 50.0
        else:
            self.max_capacity = max_capacity

        if min_speed is None:
            self.min_speed = 0.5
        else:            
            self.min_speed = min_speed

        if max_speed is None:
            self.max_speed = 1.5
        else:
            self.max_speed = max_speed

        if batch_size is not None:
            self.batch_size = torch.Size([batch_size])

        if device is not None:
            self.device = device


        if sample_type == "random":
            instance = self.random_sample_instance(
                                    num_agents=self.num_agents,
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
                                    seed=seed,
                                    device=device,
                                )

        elif sample_type == "saved":
            instance = self.get_instance(instance_name, num_agents=num_agents)

        else:
            raise ValueError("sample_type deve ser 'random' ou 'saved'")

        return instance
