import torch
from tensordict import TensorDict

import os
from os import path

from typing import Optional, Dict

import numpy as np

from maenvs4vrp.core.env_generator_builder import InstanceBuilder

from huggingface_hub import hf_hub_download
import shutil

import logging

BENCHMARK_INSTANCES_PATH = "mtvrp/data/benchmark"

VARIANT_PRESETS = [
    "cvrp",
    "ovrp",
    "ovrpb",
    "ovrpbl",
    "ovrpbltw",
    "ovrpbtw",
    "ovrpl",
    "ovrpltw",
    "ovrpmb",
    "ovrpmbl",
    "ovrpmbltw",
    "ovrpmbtw",
    "ovrptw",
    "vrpb",
    "vrpbl",
    "vrpbltw",
    "vrpbtw",
    "vrpl",
    "vrpltw",
    "vrpmb",
    "vrpmbl",
    "vrpmbltw",
    "vrpmbtw",
    "vrptw",
]

log = logging.getLogger(__name__)


class BenchmarkInstanceGenerator(InstanceBuilder):
    """
    MTVRP benchmark instance generation class.
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

        dataset = ["50_test", "100_test", "50_validation", "100_validation"]
        base_dir = path.dirname(os.path.dirname(os.path.abspath(__file__)))
        inst_dic = {}
        for pset in dataset:
            data_files = []
            numb, settype = pset.split("_")
            for problem in VARIANT_PRESETS:
                full_dir = path.join(base_dir, BENCHMARK_INSTANCES_PATH, problem)

                data_dir = os.listdir(os.path.join(full_dir, settype))
                data_files += [
                    BENCHMARK_INSTANCES_PATH
                    + "/"
                    + problem
                    + "/"
                    + settype
                    + "/"
                    + item.split(".")[0]
                    for item in data_dir
                    if numb in item
                ]

            inst_dic[pset] = data_files
        return inst_dic

    def check_instance_folders(base_dir, env):
        base_benchmark_dir = os.path.join(base_dir, env, "data/benchmark")

        for variant in VARIANT_PRESETS:
            for instance_type in ["validation", "test"]:
                for instance_name in ["100.npz", "50.npz"]:
                    file_path = os.path.join(
                        base_benchmark_dir, variant, instance_type, instance_name
                    )

                    if not os.path.isfile(file_path):
                        log.warning(
                            f"Missing instance: {file_path}. If you wish to re-download the original benchmark files, delete your {env}/data/benchmark folder and instantiate the BenchmarkInstanceGenerator class again."
                        )

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
        env = "mtvrp"
        direct = "data/benchmark"
        directory_to_be_created = os.path.join(base_dir, env, direct)

        if os.path.isdir(directory_to_be_created):
            cls.check_instance_folders(base_dir, env)

        if not (os.path.isdir(directory_to_be_created)):
            os.makedirs(directory_to_be_created)

            log.warning(
                f"Downloading benchmark files from HuggingFace to {directory_to_be_created}..."
            )

            for variant in VARIANT_PRESETS:
                for instance_type in ["val", "test"]:
                    for instance_name in ["100.npz", "50.npz"]:
                        fname = f"data/{variant}/{instance_type}/{instance_name}"

                        file_path = hf_hub_download(
                            repo_id="ai4co/routefinder",
                            repo_type="dataset",
                            filename=fname,
                        )

                        instance_type_2 = (
                            "validation" if instance_type == "val" else "test"
                        )

                        full_directory = os.path.join(
                            base_dir, env, f"data/benchmark/{variant}/{instance_type_2}"
                        )
                        if not (os.path.isdir(full_directory)):
                            os.makedirs(full_directory)
                        shutil.copy(file_path, full_directory)

    def __init__(
        self,
        problem_type: set = "all",
        instance_type: str = None,
        set_of_instances: set = None,
        device: Optional[str] = "cpu",
        batch_size: Optional[torch.Size] = 1000,
        seed: int = None,
    ) -> None:
        """
        Constructor. Create an instance space of one or several sets of data.

        Args:
            problem_type(set): Problem type. Defaults to "all".
            instance_type(str): Instance type. It must be "50_test", "100_test", "50_validation" or "100_validation". Defaults to None.
            set_of_instances(set): Set of instances paths. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".
            batch_size(torch.Size, optional): Batch size. If not specified, defaults to 1000.
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            None.
        """

        self.download_and_copy_instances()  # If instances are not on local machine, they'll be downloaded from RouteFinder's HuggingFace

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

        if problem_type is None or "all":
            problem_type = VARIANT_PRESETS
        assert (
            problem_type is not None and len(problem_type) > 0
        ), "Set of problem variants is not > 0."
        assert all(
            item in VARIANT_PRESETS for item in problem_type
        ), "Invalid variant preset."
        assert (
            instance_type in ["50_test", "100_test", "50_validation", "100_validation"]
            or instance_type is None
            or instance_type == ""
        ), "Instance type must be '50_test', '100_test','50_validation', '100_validation'."
        assert len(set_of_instances) > 0, "Set of instances not > 0."

        if set_of_instances:
            self.problem_type = problem_type
            self.instance_type = instance_type
            self.set_of_instances = set_of_instances
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
        file_path = "{path_to_generated_instances}/{instance}.npz".format(
            path_to_generated_instances=base_dir, instance=instance_name
        )

        instance = dict()
        instance["name"] = instance_name

        loaded_data = np.load(file_path)
        np_instance = {key: loaded_data[key] for key in loaded_data.files}

        data = TensorDict({}, batch_size=self.batch_size, device=self.device)
        for key in np_instance:
            data[key] = torch.from_numpy(np_instance[key])

        instance["num_agents"] = 1
        instance["num_nodes"] = data["locs"].shape[1]

        num_agents = instance["num_agents"]
        num_nodes = instance["num_nodes"]

        batch_size = data["locs"].shape[0]
        instance["batch_size"] = batch_size

        instance["name"] = instance["name"] + "_samp"

        new_data = TensorDict({}, batch_size=self.batch_size, device=self.device)

        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(-1)

        new_data["coords"] = data["locs"]  # There're always coords
        zeros = torch.zeros(
            (*self.batch_size, 1), dtype=torch.int64, device=self.device
        )
        new_data["linehaul_demands"] = torch.concat(
            [zeros, data["demand_linehaul"]], dim=1
        )  # There're always linehauls
        new_data["capacity"] = data["vehicle_capacity"]  # There're always capacities
        self.depot_idx = 0
        new_data["depot_idx"] = self.depot_idx * torch.ones(
            (*self.batch_size, 1), dtype=torch.int64, device=self.device
        )
        new_data["speed"] = data["speed"]  # There's always speeed etc.
        if "demand_backhaul" in data.keys():
            zeros = torch.zeros(
                (*self.batch_size, 1), dtype=torch.int64, device=self.device
            )
            new_data["backhaul_demands"] = torch.concat(
                [zeros, data["demand_backhaul"]], dim=1
            )
        else:
            new_data["backhaul_demands"] = torch.zeros(
                (*self.batch_size, num_nodes), dtype=torch.float32, device=self.device
            )
        if "backhaul_class" in data.keys():
            new_data["backhaul_class"] = data["backhaul_class"]
        if "time_windows" in data.keys():
            new_data["time_windows"] = data["time_windows"]
        else:
            new_data["time_windows"] = torch.zeros(
                (*self.batch_size, num_nodes, 2),
                dtype=torch.float32,
                device=self.device,
            )
            new_data["time_windows"][:, :, 1] = float("inf")
        if "service_time" in data.keys():
            new_data["service_time"] = data["service_time"]
        if "distance_limit" in data.keys():
            new_data["distance_limits"] = data["distance_limit"]
        else:
            new_data["distance_limits"] = torch.full(
                (*self.batch_size, 1), float("inf")
            )
        if "open_route" in data.keys():
            new_data["open_routes"] = data["open_route"]
        else:
            new_data["open_routes"] = torch.zeros(
                (*self.batch_size, 1), dtype=torch.bool, device=self.device
            )
        if "time_windows" in data.keys():
            new_data["end_time"] = (
                data["time_windows"][:, :, 1]
                .gather(
                    1,
                    torch.zeros(
                        (*self.batch_size, 1), dtype=torch.int64, device=self.device
                    ),
                )
                .squeeze(-1)
            )
            new_data["start_time"] = (
                data["time_windows"][:, :, 0]
                .gather(
                    1,
                    torch.zeros(
                        (*self.batch_size, 1), dtype=torch.int64, device=self.device
                    ),
                )
                .squeeze(-1)
            )
            new_data["tw_low"] = data["time_windows"][:, :, 0]
            new_data["tw_high"] = data["time_windows"][:, :, 1]
        else:
            new_data["end_time"] = torch.full((*self.batch_size, 1), float("inf"))
            new_data["start_time"] = torch.zeros(
                (*self.batch_size, 1), dtype=torch.int64, device=self.device
            )
            new_data["tw_low"] = torch.zeros(
                (*self.batch_size, num_nodes), dtype=torch.float32, device=self.device
            )
            new_data["tw_high"] = torch.full(
                (*self.batch_size, num_nodes), float("inf")
            )

        new_data["is_depot"] = torch.zeros(
            (*self.batch_size, num_nodes), dtype=torch.bool, device=self.device
        )
        new_data["is_depot"][:, self.depot_idx] = True

        instance["data"] = new_data

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

    def random_sample_instance(
        self,
        instance_name: str = None,
        num_agents: int = None,
        num_nodes: int = None,
        min_coords: float = None,
        max_coords: float = None,
        capacity: int = None,
        service_time: float = None,
        min_demands: int = None,
        max_demands: int = None,
        min_backhaul: int = None,
        max_backhaul: int = None,
        max_time: float = None,
        backhaul_ratio: float = None,
        backhaul_class: int = None,
        sample_backhaul_class: bool = False,
        max_distance_limit: float = None,
        speed: float = None,
        subsample: bool = True,
        variant_preset=None,
        use_combinations: bool = False,
        force_visit: bool = True,
        batch_size: Optional[torch.Size] = None,
        seed: int = None,
        device: Optional[str] = None,
    ) -> Dict:
        """
        Sample one instance from instance space, randomly adjusting the nodes.

        Args:
            instance_name(str): Instance file path. Defaults to None.
            num_agents(int): Total number of agents. Defaults to None.
            num_nodes(int): Total number of nodes. Defaults to None.
            min_coords(float): Minimum number of coords. Defaults to None.
            max_coords(float): Maximum number of coords. Defaults to None.
            capacity(int): Vehicles' capacity. Defaults to None.
            service_time(float): Service time. Defaults to None.
            min_demands(int): Minimum number of demands. Defaults to None.
            max_demands(int): Maximum number of demands. Defaults to None.
            min_backhaul(int): Minimum number of backhauls. Defaults to None.
            max_backhaul(int): Maximum number of backhauls. Defaults to None.
            max_time(float): Maximum route time. Defaults to None.
            backhaul_ratio(float): Ratio of backhaul demands. Defaults to None.
            backhaul_class(int): Class of backhaul problem. If 1, it's unmixed, if 2, it's mixed. Defaults to None.
            sample_backhaul_class(bool): If backhaul class is sampled across batches. Defaults to False.
            max_distance_limit(float): Route distance limits. Defaults to None.
            speed(float): Vehicles' speed. Defaults to None.
            subsample(bool): If problem variants are to be sampled. Defaults to True.
            variant_preset(str): Variant preset to be sampled. Defaults to None.
            use_combinations(bool): It considers combinations for which sampling mask the instance is defined. Defaults to False.
            force_visit(bool): It forces the agent to visit all feasible nodes before going back to depot. Defaults to True.
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".

        Returns:
            Dict: Instance data.
        """
        if seed is not None:
            self._set_seed(seed)

        new_instance = dict()
        instance = self.get_instance(instance_name, num_agents)

        new_instance["num_agents"] = instance["num_agents"]

        if num_nodes is not None:
            num_nodes = min(num_nodes, instance["num_nodes"])
            new_instance["num_nodes"] = num_nodes
        else:
            num_nodes = instance["num_nodes"]
            new_instance["num_nodes"] = instance["num_nodes"]

        batch_size = instance["batch_size"]

        idxs = torch.arange(0, num_nodes, device=self.device).expand(
            batch_size, num_nodes
        )
        depots = idxs[:, 0:1]
        non_depots = idxs[:, 1:]
        indices = torch.argsort(torch.rand(*non_depots.shape), dim=-1)
        index = torch.cat([depots, indices], dim=1)
        index = index[:, :num_nodes]

        new_data = TensorDict({}, batch_size=self.batch_size, device=self.device)

        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(-1)

        data = instance["data"]

        new_data["coords"] = data["coords"][batch_idx, index]  # There're always coords
        new_data["linehaul_demands"] = data["linehaul_demands"][
            batch_idx, index
        ]  # There're always linehauls
        new_data["capacity"] = data["capacity"]  # There're always capacities
        self.depot_idx = 0
        new_data["depot_idx"] = self.depot_idx * torch.ones(
            (*self.batch_size, 1), dtype=torch.int64, device=self.device
        )
        new_data["speed"] = data["speed"]  # There's always speeed etc.
        if "backhaul_demands" in data.keys():
            new_data["backhaul_demands"] = data["backhaul_demands"][batch_idx, index]
        if "backhaul_class" in data.keys():
            new_data["backhaul_class"] = data["backhaul_class"]
        if "time_windows" in data.keys():
            new_data["time_windows"] = data["time_windows"][batch_idx, index]
        if "service_time" in data.keys():
            new_data["service_time"] = data["service_time"][batch_idx, index]
        if "distance_limits" in data.keys():
            new_data["distance_limits"] = data["distance_limits"]
        else:
            new_data["distance_limits"] = torch.full(
                (*self.batch_size, 1), float("inf")
            )
        if "open_routes" in data.keys():
            new_data["open_routes"] = data["open_routes"]
        if "time_windows" in data.keys():
            new_data["end_time"] = (
                data["time_windows"][:, :, 1]
                .gather(
                    1,
                    torch.zeros(
                        (*self.batch_size, 1), dtype=torch.int64, device=self.device
                    ),
                )
                .squeeze(-1)
            )
            new_data["start_time"] = (
                data["time_windows"][:, :, 0]
                .gather(
                    1,
                    torch.zeros(
                        (*self.batch_size, 1), dtype=torch.int64, device=self.device
                    ),
                )
                .squeeze(-1)
            )
            new_data["tw_low"] = new_data["time_windows"][:, :, 0]
            new_data["tw_high"] = new_data["time_windows"][:, :, 1]
        else:
            new_data["end_time"] = torch.full((*self.batch_size, 1), float("inf"))
            new_data["start_time"] = torch.zeros(
                (*self.batch_size, 1), dtype=torch.int64, device=self.device
            )
            new_data["tw_low"] = torch.zeros(
                (*self.batch_size, num_nodes), dtype=torch.float32, device=self.device
            )
            new_data["tw_high"] = torch.full(
                (*self.batch_size, num_nodes), float("inf")
            )

        new_data["is_depot"] = torch.zeros(
            (*self.batch_size, num_nodes), dtype=torch.bool, device=self.device
        )
        new_data["is_depot"][:, self.depot_idx] = True

        new_instance["data"] = new_data

        return new_instance

    def sample_name_from_set(self, seed: int = None) -> str:
        """
        Sample one instance from instance set.

        Args:
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            str: Instance sample name.
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
        sample_type: str = "random",
        instance_name: str = None,
        num_agents: int = None,
        num_nodes: int = None,
        min_coords: float = None,
        max_coords: float = None,
        capacity: int = None,
        service_time: float = None,
        min_demands: int = None,
        max_demands: int = None,
        min_backhaul: int = None,
        max_backhaul: int = None,
        max_time: float = None,
        backhaul_ratio: float = None,
        backhaul_class: int = None,
        sample_backhaul_class: bool = False,
        max_distance_limit: float = None,
        speed: float = None,
        subsample: bool = True,
        variant_preset=None,
        use_combinations: bool = False,
        force_visit: bool = True,
        batch_size: Optional[torch.Size] = None,
        seed: int = None,
        n_augment: Optional[int] = None,
        device: Optional[str] = None,
    ) -> Dict:
        """
        Sample one instance from instance space.

        Args:
            sample_type(str): Type of instance to sample. It can be "random" or "augment". Defaults to "random".
            instance_name(str): Instance file path. Defaults to None.
            num_agents(int): Total number of agents. Defaults to None.
            num_nodes(int): Total number of nodes. Defaults to None.
            min_coords(float): Minimum number of coords. Defaults to None.
            max_coords(float): Maximum number of coords. Defaults to None.
            capacity(int): Vehicles' capacity. Defaults to None.
            service_time(float): Service time. Defaults to None.
            min_demands(int): Minimum number of demands. Defaults to None.
            max_demands(int): Maximum number of demands. Defaults to None.
            min_backhaul(int): Minimum number of backhauls. Defaults to None.
            max_backhaul(int): Maximum number of backhauls. Defaults to None.
            max_time(float): Maximum route time. Defaults to None.
            backhaul_ratio(float): Ratio of backhaul demands. Defaults to None.
            backhaul_class(int): Class of backhaul problem. If 1, it's unmixed, if 2, it's mixed. Defaults to None.
            sample_backhaul_class(bool): If backhaul class is sampled across batches. Defaults to False.
            max_distance_limit(float): Route distance limits. Defaults to None.
            speed(float): Vehicles' speed. Defaults to None.
            subsample(bool): If problem variants are to be sampled. Defaults to True.
            variant_preset(str): Variant preset to be sampled. Defaults to None.
            use_combinations(bool): It considers combinations for which sampling mask the instance is defined. Defaults to False.
            force_visit(bool): It forces the agent to visit all feasible nodes before going back to depot. Defaults to True.
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.
            n_augment(int, optional): Number of augmentations. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".

        Returns:
            Dict: Instance data.
        """
        if seed is not None:
            self._set_seed(seed)

        if instance_name == None:
            instance_name = self.sample_name_from_set(seed=seed)
        else:
            instance_name = instance_name

        if num_agents is None:
            self.num_agents = 2
        else:
            self.num_agents = num_agents

        if num_nodes is None:
            self.num_nodes = 15
        else:
            self.num_nodes = num_nodes

        if min_coords is None:
            self.min_coords = 0.0
        else:
            self.min_coords = min_coords

        if max_coords is None:
            self.max_coords = 1.0
        else:
            self.max_coords = max_coords

        if capacity is None:
            self.capacity = 50.0
        else:
            self.capacity = capacity

        if service_time is None:
            self.service_time = 0.2
        else:
            self.service_time = service_time

        if min_demands is None:
            self.min_demands = 1
        else:
            self.min_demands = min_demands

        if max_demands is None:
            self.max_demands = 10
        else:
            self.max_demands = max_demands

        if min_backhaul is None:
            self.min_backhaul = 1
        else:
            self.min_backhaul = min_backhaul

        if max_backhaul is None:
            self.max_backhaul = 10
        else:
            self.max_backhaul = max_backhaul

        if max_time is None:
            self.max_time = 4.6
        else:
            self.max_time = max_time

        if backhaul_ratio is None:
            self.backhaul_ratio = 0.2
        else:
            self.backhaul_ratio = backhaul_ratio

        if backhaul_class is None:
            self.backhaul_class = 1
        else:
            self.backhaul_class = backhaul_class

        if sample_backhaul_class is None:
            self.sample_backhaul_class = False
        else:
            self.sample_backhaul_class = sample_backhaul_class

        if max_distance_limit is None:
            self.max_distance_limit = 2.8
        else:
            self.max_distance_limit = max_distance_limit

        if speed is None:
            self.speed = 1.0
        else:
            self.speed = speed

        if sample_type == "random":
            instance = self.random_sample_instance(
                instance_name=instance_name,
                num_agents=num_agents,
                num_nodes=num_nodes,
                min_coords=min_coords,
                max_coords=max_coords,
                capacity=capacity,
                service_time=service_time,
                min_demands=min_demands,
                max_demands=max_demands,
                min_backhaul=min_backhaul,
                max_backhaul=max_backhaul,
                max_time=max_time,
                backhaul_ratio=backhaul_ratio,
                backhaul_class=backhaul_class,
                sample_backhaul_class=sample_backhaul_class,
                max_distance_limit=max_distance_limit,
                speed=speed,
                subsample=subsample,
                variant_preset=variant_preset,
                use_combinations=use_combinations,
                force_visit=force_visit,
                batch_size=batch_size,
                seed=seed,
                device=device,
            )
        else:
            instance = self.get_instance(instance_name, num_agents=num_agents)

        return instance
