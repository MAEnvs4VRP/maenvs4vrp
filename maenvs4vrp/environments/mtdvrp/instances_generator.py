"""
Adapted from: https://github.com/ai4co/routefinder/blob/main/routefinder/envs/mtvrp/generator.py 

"""


import torch
from torch import Tensor
from tensordict import TensorDict

from typing import Optional, Union, Callable, Dict, Tuple

from maenvs4vrp.core.env_generator_builder import InstanceBuilder
from maenvs4vrp.utils.ops import get_distance

from torch.distributions import Uniform

import os
import pickle

GENERATED_INSTANCES_PATH = 'gmtdvrp/data/generated'

def get_vehicle_capacity(num_loc: int) -> int:
    """
    Get vehicle capacity.

    Args:
        num_loc(int): Number of nodes.

    Returns:
        capacity(int): Vehicle capacity.
    """
    if num_loc > 1000:
        extra_cap = 1000 // 5 + (num_loc - 1000) // 33.3
    elif num_loc > 20:
        extra_cap = num_loc // 5
    else:
        extra_cap = 0
    return 30 + extra_cap

VARIANT_PRESETS = [
    'cvrp', 'ovrp', 'ovrpb', 'ovrpbl', 'ovrpbltw', 'ovrpbtw',
    'ovrpl', 'ovrpltw', 'ovrpmb', 'ovrpmbl', 'ovrpmbltw', 'ovrpmbtw',
    'ovrptw', 'vrpb', 'vrpbl', 'vrpbltw', 'vrpbtw', 'vrpl',
    'vrpltw', 'vrpmb', 'vrpmbl', 'vrpmbltw', 'vrpmbtw', 'vrptw'
    ]

VARIANT_PRESETS_UNMIXED = [
    'cvrp', 'ovrp', 'ovrpb', 'ovrpbl', 'ovrpbltw', 'ovrpbtw',
    'ovrpl', 'ovrpltw',
    'ovrptw', 'vrpb', 'vrpbl', 'vrpbltw', 'vrpbtw', 'vrpl',
    'vrpltw', 'vrptw'
    ]

VARIANT_PRESETS_MIXED = [
    'ovrpmb', 'ovrpmbl', 'ovrpmbltw', 'ovrpmbtw',
    'vrpmb', 'vrpmbl', 'vrpmbltw', 'vrpmbtw'
]

VARIANT_PROBS_PRESETS = { #Variant Probabilities
        "all": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5},
        "single_feat": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5},
        "single_feat_otw": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5, "OTW": 0.5},
        "cvrp": {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 0.0},
        "ovrp": {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 0.0},
        "vrpb": {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 1.0},
        "vrpl": {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 0.0},
        "vrptw": {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 0.0},
        "ovrptw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 0.0},
        "ovrpb": {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 1.0},
        "ovrpl": {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 0.0},
        "vrpbl": {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 1.0},
        "vrpbtw": {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 1.0},
        "vrpltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 0.0},
        "ovrpbl": {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 1.0},
        "ovrpbtw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 1.0},
        "ovrpltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 0.0},
        "vrpbltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 1.0},
        "ovrpbltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 1.0},
    }

class InstanceGenerator(InstanceBuilder):

    """
    MTDVRP instance generation class.
    """

    @classmethod
    def get_list_of_benchmark_instances(cls, mixed: bool = True):

        """
        Get list of generated instances.

        Args:
            mixed(bool): If True, it gets all instances. If False, it gets only unmixed instances. Defaults to True.

        Returns:
            benchmark_instances(list): Generated instances.
        """

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        benchmark_instances = {}

        generated = os.listdir(os.path.join(base_dir, GENERATED_INSTANCES_PATH))
        
        for folder in generated:
            benchmark_instances[folder] = {}
            benchmark_instances[folder]['validation'] = []
            benchmark_instances[folder]['test'] = []

            if mixed:
            
                for problem_type in VARIANT_PRESETS:
                    val_path = os.path.join(GENERATED_INSTANCES_PATH, folder, problem_type, 'validation')
                    test_path = os.path.join(GENERATED_INSTANCES_PATH, folder, problem_type, 'test')

                    for s in os.listdir(os.path.join(base_dir, val_path)):
                        val = val_path + '/' + s.split('.')[0]
                        benchmark_instances[folder]['validation'].append(val)

                    for s in os.listdir(os.path.join(base_dir, test_path)):
                        test = test_path + '/' + s.split('.')[0]
                        benchmark_instances[folder]['test'].append(test)

            else:

                for problem_type in VARIANT_PRESETS_UNMIXED:
                    val_path = os.path.join(GENERATED_INSTANCES_PATH, folder, problem_type, 'validation')
                    test_path = os.path.join(GENERATED_INSTANCES_PATH, folder, problem_type, 'test')

                    for s in os.listdir(os.path.join(base_dir, val_path)):
                        val = val_path + '/' + s.split('.')[0]
                        benchmark_instances[folder]['validation'].append(val)

                    for s in os.listdir(os.path.join(base_dir, test_path)):
                        test = test_path + '/' + s.split('.')[0]
                        benchmark_instances[folder]['test'].append(test)
    
        return benchmark_instances


    def __init__(
        self,
        instance_type:str = 'validation',
        set_of_instances:set = None,
        device: Optional[str] = "cpu",
        batch_size: Optional[torch.Size] = None,
        seed: int = None
    ) -> None:

        """
        Constructor. Instance generator.

        Args:       
            instance_type(str): Instance type. Can be "validation" or "test". Defaults to "validation".
            set_of_instances(set):  Set of instances file names. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".
            batch_size(torch.Size, optional): Batch size. If not specified, defaults to 1.
            seed(int): Random number generator seed. Defaults to None.

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

        assert instance_type in ['test', 'validation'] or instance_type is None or instance_type == '', f"Instance type must be 'test', 'validation', '' or None." #If None or empty, it loads both test and validation
        self.set_of_instances = set_of_instances

        if set_of_instances:
            self.instance_type = instance_type
            self.load_set_of_instances()

    def load_set_of_instances(
        self,
        set_of_instances:set = None
    ):
        
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
    
    def read_instance_data(self, instance_name:str):
        
        """
        Read instance data from file.

        Args:
            instance_name(str): Instance file name.

        Returns: 
            Dict: Instance data.
        """
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        generated_file = '{path_to_generated_instances}/{instance}.pkl' \
                        .format(path_to_generated_instances=base_dir,
                                instance=instance_name)
        with open(generated_file, 'rb') as fp:
            instance = pickle.load(fp)
        self.batch_size = instance['data'].batch_size
        instance['data'] = instance['data'].to(self.device)
        return instance
    
    def get_instance(self, instance_name:str, num_agents:int=None) -> Dict:
        
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
            assert num_agents>0, f"number of agents must be grater them 0!"
            instance['num_agents'] = num_agents

        return instance

    @staticmethod
    def _default_open(td, remove):
        td['open_routes'][remove] = False
        return td

    @staticmethod
    def _default_time_windows(td, remove):
        default_tw = torch.zeros_like(td['time_windows'])
        default_tw[..., 1] = float('inf')
        td['time_windows'][remove] = default_tw[remove]
        td['service_time'][remove] = torch.zeros_like(td['service_time'][remove])
        return td
    
    @staticmethod
    def _default_distance_limit(td, remove):
        td['distance_limits'][remove] = float('inf')
        return td
    
    @staticmethod
    def _default_backhaul(td, remove):
        td['linehaul_demands'][remove] = (
            td['linehaul_demands'][remove] + td['backhaul_demands'][remove]
        )
        td['backhaul_demands'][remove] = 0
        return td
    
    def generate_demands(self, batch_size: int, num_nodes: int) -> torch.Tensor:
        """
        Generate demands.

        Args:
            batch_size(int): Batch size.
            num_nodes(int): Number of nodes.

        Returns:
            torch.Tensor: Linehaul and backhaul demands.
        """
        linehaul_demand = torch.FloatTensor(*batch_size, num_nodes).uniform_(
            self.min_demands - 1, self.max_demands - 1
        )
        linehaul_demand = (linehaul_demand.int() + 1).float()
        # Backhaul demand sampling
        backhaul_demand = torch.FloatTensor(*batch_size, num_nodes).uniform_(
            self.min_backhaul - 1, self.max_backhaul - 1
        )
        backhaul_demand = (backhaul_demand.int() + 1).float()
        is_linehaul = torch.rand(*batch_size, num_nodes) > self.backhaul_ratio
        backhaul_demand = (
            backhaul_demand * ~is_linehaul
        )  # keep only values where they are not linehauls
        linehaul_demand = linehaul_demand * is_linehaul
        return linehaul_demand, backhaul_demand
    
    def generate_backhaul_class(self, shape: Tuple[int, int], sample: bool = False):
        """
        Generate backhaul class.

        Args:
            shape(Tuple): Tensor shape.
            sample(bool): Sample backhaul class. Defaults to False.

        Returns:
            torch.Tensor: Linehaul and backhaul demands.
        """
        if sample:
            return torch.randint(1, 3, shape, dtype=torch.float32)
        else:
            return torch.full(shape, self.backhaul_class, dtype=torch.float32)
        
    def generate_distance_limit(
        self, shape: Tuple[int, int], coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate distance limits.

        Args:
            shape(Tuple): Tensor shape.
            coords(torch.Tensor): Nodes coordinates.

        Returns:
            torch.Tensor: Distance limits.
        """
        max_dist = torch.max(torch.cdist(coords[:, 0:1], coords[:, 1:]).squeeze(-2), dim=1)[0]
        dist_lower_bound = 2 * max_dist + 1e-6
        max_distance_limit = torch.maximum(
            torch.full_like(dist_lower_bound, self.max_distance_limit),
            dist_lower_bound + 1e-6,
        )

        # We need to sample from the `distribution` module to get the same distribution with a tensor as input
        return torch.distributions.Uniform(dist_lower_bound, max_distance_limit).sample()[
            ..., None
        ]
    
    def generate_time_windows(
        self,
        coords: torch.Tensor = None,
        speed: torch.Tensor = None,
    ) -> torch.Tensor:
        
        """
        Generate time windows.

        Args:
            coords(torch.Tensor): Nodes coordinates.
            speed(torch.Tensor): Agents speed.

        Returns:
            torch.Tensor: Time windows and service times.
        """
        
        batch_size, n_loc = coords.shape[0], coords.shape[1] - self.num_depots  # no depot

        a, b, c = 0.15, 0.18, 0.2
        service_time = a + (b - a) * torch.rand(batch_size, n_loc)
        tw_length = b + (c - b) * torch.rand(batch_size, n_loc)

        # Expand dimensions to make them compatible for broadcasting
        tensor_a = coords[:, 0 : self.num_depots]  # Shape: [B, num_depots, 2]
        tensor_b = coords[:, self.num_depots :]  # Shape: [B, N, 2]

        tensor_a_expanded = tensor_a[:, :, None, :]  # Shape: [B, num_depots, 1, 2]
        tensor_b_expanded = tensor_b[:, None, :, :]  # Shape: [B, 1, N, 2]

        d_0i = get_distance(
            tensor_a_expanded, tensor_b_expanded
        )  # Shape: [B, num_depots, N]

        d_0i, _ = torch.max(d_0i, dim=1)  # Shape: [B, N]

        # d_0i = get_distance(locs[:, 0 : 1], locs[:, self.num_depots :]) # old version with 1 depot

        h_max = (self.max_time - service_time - tw_length) / d_0i * speed - 1
        tw_start = (1 + (h_max - 1) * torch.rand(batch_size, n_loc)) * d_0i / speed
        tw_end = tw_start + tw_length

        # Depot tw is [0, max_time]
        time_windows = torch.stack(
            (
                torch.cat(
                    (torch.zeros(batch_size, self.num_depots), tw_start), dim=-1
                ),  # start
                torch.cat(
                    (torch.full((batch_size, self.num_depots), self.max_time), tw_end),
                    dim=-1,
                ),
            ),  # end
            dim=-1,
        )
        # depot service time is 0
        service_time = torch.cat(
            (torch.zeros(batch_size, self.num_depots), service_time), dim=-1
        )
        return time_windows, service_time  # [B, num_depots + N, 2], [B, num_depots + N]
    
    def subsample_variant(
        self,
        prob_open_routes: float = 0.5,
        prob_time_windows: float = 0.5,
        prob_limit: float = 0.5,
        prob_backhaul: float = 0.5,
        td: TensorDict = None,
        variant_preset = None,
    ) -> torch.Tensor:

        """
        Subsample variant. If variant_preset is specified, it loads that variant. Otherwise it samples variant's parameters across batches based on probabilities.

        Args:
            prob_open_routes(float): Probability of open routes. Defaults to 0.5.
            prob_time_windows(float): Probability of time windows. Defaults to 0.5.
            prob_limit(float): Probability of distance limits. Defaults to 0.5.
            prob_backhaul(float): Probability of backhaul. Defaults to 0.5.
            td(TensorDict): Environment instance tensor. Defaults to None.
            variant_preset(TensorDict): Variant preset. Defaults to None.

        Returns:
            td(TensorDict): Environment instance tensor.
        """
        
        td['has_open_routes'] = torch.zeros((*self.batch_size, 1), dtype=torch.bool)
        td['has_time_windows'] = torch.zeros((*self.batch_size, 1), dtype=torch.bool)
        td['has_distance_limits'] = torch.zeros((*self.batch_size, 1), dtype=torch.bool)
        td['has_backhauls'] = torch.zeros((*self.batch_size, 1), dtype=torch.bool)
        
        if variant_preset is not None:
            variant_probs = VARIANT_PROBS_PRESETS.get(variant_preset)
            assert variant_probs is not None, f"Variant preset {variant_preset} not found! \
                                                Avaliable presets are {VARIANT_PROBS_PRESETS.keys()} with probabilities {VARIANT_PROBS_PRESETS.values()}"
        else:
            variant_probs = {
                "O": prob_open_routes,
                "TW": prob_time_windows,
                "L": prob_limit,
                "B": prob_backhaul
            }

        for key, prob in variant_probs.items():
            assert 0 <= prob <= 1, f"Probability {key} must be between 0 and 1"

        self.variant_probs = variant_probs
        self.variant_preset = variant_preset

        variant_probs = torch.Tensor(list(self.variant_probs.values())) #Convert dict into tensor

        if self.use_combinations:
            keep_mask = torch.rand(*self.batch_size, 4) >= variant_probs #O, TW, L, B
            td['has_open_routes'][keep_mask[:, 0]] = True
            td['has_time_windows'][keep_mask[:, 1]] = True
            td['has_distance_limits'][keep_mask[:, 2]] = True
            td['has_backhauls'][keep_mask[:, 3]] = True
        
        else:
            if self.variant_preset in list(VARIANT_PROBS_PRESETS.keys()) and self.variant_preset not in ("all", "cvrp", "single_feat", "single_feat_otw"):
                cvrp_prob = 0
            else:
                cvrp_prob = 0.5

            if self.variant_preset in ("all", "cvrp", "single_feat", "single_feat_otw"):
                indexes = torch.distributions.Categorical(
                    torch.Tensor(list(self.variant_probs.values()) + [cvrp_prob])[
                        None
                    ].repeat(*self.batch_size, 1)
                ).sample()

                if self.variant_preset == "single_feat_otw":
                    keep_mask = torch.zeros((*self.batch_size, 6), dtype=torch.bool)
                    keep_mask[torch.arange(*self.batch_size), indexes] = True

                    keep_mask[:, :2] |= keep_mask[:, 4:5]

                    td['has_open_routes'][keep_mask[:, 0]] = True
                    td['has_time_windows'][keep_mask[:, 1]] = True
                    td['has_distance_limits'][keep_mask[:, 2]] = True
                    td['has_backhauls'][keep_mask[:, 3]] = True
                    td['has_open_routes'][keep_mask[:, 4]] = True
                    td['has_time_windows'][keep_mask[:, 4]] = True

                else:
                    keep_mask = torch.zeros((*self.batch_size, 5), dtype=torch.bool)
                    keep_mask[torch.arange(*self.batch_size), indexes] = True

                    td['has_open_routes'][keep_mask[:, 0]] = True
                    td['has_time_windows'][keep_mask[:, 1]] = True
                    td['has_distance_limits'][keep_mask[:, 2]] = True
                    td['has_backhauls'][keep_mask[:, 3]] = True

            else:

                keep_mask = torch.zeros((*self.batch_size, 4), dtype=torch.bool)
                indexes = torch.nonzero(variant_probs).squeeze()
                keep_mask[:, indexes] = True

                td['has_open_routes'][keep_mask[:, 0]] = True
                td['has_time_windows'][keep_mask[:, 1]] = True
                td['has_distance_limits'][keep_mask[:, 2]] = True
                td['has_backhauls'][keep_mask[:, 3]] = True
        
        td = self._default_open(td, ~keep_mask[:, 0])
        td = self._default_time_windows(td, ~keep_mask[:, 1])
        td = self._default_distance_limit(td, ~keep_mask[:, 2])
        td = self._default_backhaul(td, ~keep_mask[:, 3])

        self.keep_mask = keep_mask

        return td    
    

    def random_generate_instance(
        self,
        num_depots: int = None,
        num_agents: int = None,
        num_nodes: int = None,
        min_coords: float = None,
        max_coords: float = None,
        capacity: float = None,
        service_time: float = None,
        min_demands: int = None,
        max_demands: int = None,
        min_backhaul: int = None,
        max_backhaul: int = None,
        max_time: float = None,
        backhaul_ratio: float = None,
        backhaul_class: int = None,
        sample_backhaul_class: bool = None,
        max_distance_limit: float = None,
        speed: float = None,
        subsample: bool = True,
        variant_preset=None,
        use_combinations: bool = False,
        batch_size: Optional[torch.Size] = None,
        seed: int = None,
        device: Optional[str] = "cpu"
    ) -> TensorDict:

        """
        Generate random instance.

        Args:
            num_depots(int): Total number of depots. Defaults to None. 
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
            sample_backhaul_class(bool): If backhaul class is sampled across batches. Defaults to None.
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
            TensorDict: Instance data.
        """

        if seed is not None:
            self._set_seed(seed)

        if num_depots is not None:
            assert num_depots>0, f"Number of depots must be greater than 0!"
        if num_agents is not None:
            assert num_agents>0, f"Number of agents must be greater than 0!"
        if num_nodes is not None:
            assert num_nodes>0, f"Number of nodes must be greater than 0!"
        if capacity is not None:
            assert capacity>0, f"Capacity must be greater than 0!"
        if service_time is not None:
            assert service_time>0, f"Service times must be greater than 0!"
        if max_time is not None:
            assert max_time>0, f"Service times must be greater than 0!"
        if max_distance_limit is not None:
            assert max_distance_limit>0, f"Distance limit must be greater than 0!"
        if speed is not None:
            assert speed>0, f"Speed must be greater than 0!"
        if backhaul_class is not None:
            assert backhaul_class in (1, 2), f"Backhaul class must be in [1, 2]!"

        if batch_size is None:
            batch_size = self.batch_size
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        instance = TensorDict({}, batch_size=batch_size, device=self.device)

        #Depot generation. All 0.
        instance['num_agents'] = torch.full((*self.batch_size, 1), num_agents)
        instance['num_depots'] = torch.full((*self.batch_size, 1), num_depots)
        self.depot_idx = torch.arange(self.num_depots, device=self.device).repeat((*self.batch_size, 1))
        instance['depot_idx'] = self.depot_idx

        #Coords unfiform generation
        coords = torch.FloatTensor(*self.batch_size, num_nodes, 2).uniform_(min_coords, max_coords) #Nodes. (x,y)
        instance['coords'] = coords

        #Capacity
        vehicle_capacity = torch.full((*self.batch_size, 1), self.capacity, dtype=torch.float32)
        instance['capacity'] = vehicle_capacity
        instance['original_capacity'] = torch.full((*self.batch_size, 1), self.capacity, dtype=torch.float32)

        #Demands
        linehaul_demands, backhaul_demands = self.generate_demands(batch_size=self.batch_size, num_nodes=self.num_nodes)
        linehaul_demands[:, self.depot_idx] = 0.0
        backhaul_demands[:, self.depot_idx] = 0.0
        instance['linehaul_demands'] = linehaul_demands
        instance['backhaul_demands'] = backhaul_demands

        #Backhaul Class. If sample true it's random. Otherwise it's defined in constructor.
        backhaul_class = self.generate_backhaul_class(shape=(*self.batch_size, 1), sample=self.sample_backhaul_class)
        instance['backhaul_class'] = backhaul_class

        #Open routes
        instance['open_routes'] = torch.ones(*self.batch_size, 1, dtype=torch.bool)

        #Speed
        instance['speed'] = torch.full((*self.batch_size, 1), self.speed, dtype=torch.float32)

        #Time windows and service times
        time_windows, service_time = self.generate_time_windows(coords, self.speed)

        instance['time_windows'] = time_windows
        instance['service_time'] = service_time

        instance['tw_low'] = time_windows[:, :, 0]
        instance['tw_high'] = time_windows[:, :, 1]

        instance['is_depot'] = torch.zeros((*self.batch_size, num_nodes), dtype=torch.bool, device=self.device)
        instance['is_depot'][:, self.depot_idx] = True

        #Start time and end time
        instance['start_time'] = time_windows[:, :, 0].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                          dtype=torch.int64, device=self.device)).squeeze(-1)
        instance['end_time'] = time_windows[:, :, 1].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                        dtype=torch.int64, device=self.device)).squeeze(-1)

        #Distance limits
        distance_limits = self.generate_distance_limit(shape=(*self.batch_size, 1), coords=coords)
        instance['distance_limits'] = distance_limits

        if self.subsample:
            instance = self.subsample_variant(td=instance, variant_preset=self.variant_preset)
        
        instance_info = {'name': 'random_instance',
                         'num_depots': num_depots,
                         'num_nodes': num_nodes,
                         'num_agents': num_agents,
                         'data': instance}
        
        return instance_info

    def augment_generate_instance(
        self,
        num_depots: int = None,
        num_agents: int = None,
        num_nodes: int = None,
        min_coords: float = None,
        max_coords: float = None,
        capacity: float = None,
        service_time: float = None,
        min_demands: int = None,
        max_demands: int = None,
        min_backhaul: int = None,
        max_backhaul: int = None,
        max_time: float = None,
        backhaul_ratio: float = None,
        backhaul_class: int = None,
        sample_backhaul_class: bool = None,
        max_distance_limit: float = None,
        speed: float = None,
        subsample: bool = True,
        variant_preset=None,
        use_combinations: bool = False,
        batch_size: Optional[torch.Size] = None,
        n_augment:int = 2,
        seed: int = None,
        device: Optional[str] = "cpu"
    ) -> TensorDict:
        
        """
        Generate augmented instance.

        Args:
            num_depots(int): Total number of depots. Defaults to None.
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
            sample_backhaul_class(bool): If backhaul class is sampled across batches. Defaults to None.
            max_distance_limit(float): Route distance limits. Defaults to None.
            speed(float): Vehicles' speed. Defaults to None.
            subsample(bool): If problem variants are to be sampled. Defaults to True.
            variant_preset(str): Variant preset to be sampled. Defaults to None.
            use_combinations(bool): It considers combinations for which sampling mask the instance is defined. Defaults to False.
            force_visit(bool): It forces the agent to visit all feasible nodes before going back to depot. Defaults to True.
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int): Number of augmentations. Defaults to 2.
            seed(int): Random number generator seed. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".

        Returns:
            TensorDict: Instance data.
        """
        
        if seed is not None:
            self._set_seed(seed)

        if num_depots is not None:
            assert num_depots>0, f"Number of depots must be greater than 0!"
        if num_agents is not None:
            assert num_agents>0, f"Number of agents must be greater than 0!"
        if num_nodes is not None:
            assert num_nodes>0, f"Number of nodes must be greater than 0!"
        if capacity is not None:
            assert capacity>0, f"Capacity must be greater than 0!"
        if service_time is not None:
            assert service_time>0, f"Service times must be greater than 0!"
        if max_time is not None:
            assert max_time>0, f"Service times must be greater than 0!"
        if max_distance_limit is not None:
            assert max_distance_limit>0, f"Distance limit must be greater than 0!"
        if speed is not None:
            assert max_time>0, f"Speed must be greater than 0!"
        if backhaul_class is not None:
            assert backhaul_class in (1, 2), f"Backhaul class must be in [1, 2]!"

        assert self.batch_size.numel()%n_augment == 0, f"Batch size must be divisible by n_augment!"
        s_batch_size = self.batch_size.numel() // n_augment #Same batch size
        self.s_batch_size = torch.Size([s_batch_size])

        instance_info_s = self.random_generate_instance( #Generate random instance
            num_depots = num_depots,
            num_agents = num_agents,
            num_nodes = num_nodes,
            min_coords = min_coords,
            max_coords = max_coords,
            capacity = capacity,
            service_time = service_time,
            min_demands = min_demands,
            max_demands = max_demands,
            min_backhaul = min_backhaul,
            max_backhaul = max_backhaul,
            max_time = max_time,
            backhaul_ratio = backhaul_ratio,
            backhaul_class = backhaul_class,
            sample_backhaul_class = sample_backhaul_class,
            max_distance_limit = max_distance_limit,
            speed = speed,
            subsample = subsample,
            variant_preset = variant_preset,
            use_combinations = use_combinations,
            batch_size = self.s_batch_size,
            seed = seed,
            device = device
        )

        self.batch_size = torch.Size(batch_size)

        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)

        for key in instance_info_s['data'].keys():
            if len(instance_info_s['data'][key].shape) == 3: #3 dimension tensors
                instance[key] = instance_info_s['data'][key].repeat(n_augment, 1, 1)
            elif len(instance_info_s['data'][key].shape) == 2: #2 dimension tensors
                instance[key] = instance_info_s['data'][key].repeat(n_augment, 1)
            elif len(instance_info_s['data'][key].shape) == 1: #1 dimension tensors
                instance[key] = instance_info_s['data'][key].repeat(n_augment)

        instance_info = {'name':'augmented_instance',
                         'num_depots': num_depots,
                         'num_nodes': num_nodes,
                         'num_agents': num_agents,
                         'data':instance}
        
        return instance_info
    
    def sample_name_from_set(self, seed:int=None)-> str:
        """
        Sample one instance from instance set.

        Args:
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            str: Instance name.
        """
        if seed is not None:
            self._set_seed(seed)
        assert len(self.set_of_instances)>0, f"set_of_instances has to have at least one instance!"

        return list(self.set_of_instances)[torch.randint(0, len(self.set_of_instances), (1,)).item()]
    
    def sample_instance(
        self,
        num_depots: int = 3,
        num_agents: int = 2,
        num_nodes: int = 15,
        min_coords: float = 0.0,
        max_coords: float = 1.0,
        capacity: float = None,
        service_time: float = 0.2,
        min_demands: int = 1,
        max_demands: int = 10,
        min_backhaul: int = 1,
        max_backhaul: int = 10,
        max_time: float = 4.6,
        backhaul_ratio: float = 0.2,
        backhaul_class: int = 1,
        sample_backhaul_class: bool = False,
        max_distance_limit: float = 2.8,
        speed: float = 1.0,
        subsample: bool = True,
        variant_preset='all',
        use_combinations: bool = False,
        batch_size: Optional[torch.Size] = None,
        n_augment: Optional[int] = 2,
        sample_type: str = 'random',
        instance_name: str = None,
        seed: int = None,
        device: Optional[str] = "cpu"
    ):
        
        """
        Sample one instance from instance space.

        Args:
            num_depots(int): Total number of depots. Defaults to 3.
            num_agents(int): Total number of agents. Defaults to 2.
            num_nodes(int): Total number of nodes. Defaults to 15.
            min_coords(float): Minimum number of coords. Defaults to 0.0.
            max_coords(float): Maximum number of coords. Defaults to 1.0.
            capacity(int): Vehicles' capacity. Defaults to None.
            service_time(float): Service time. Defaults to 0.2.
            min_demands(int): Minimum number of demands. Defaults to 1.
            max_demands(int): Maximum number of demands. Defaults to 10.
            min_backhaul(int): Minimum number of backhauls. Defaults to 1.
            max_backhaul(int): Maximum number of backhauls. Defaults to 10.
            max_time(float): Maximum route time. Defaults to 4.6.
            backhaul_ratio(float): Ratio of backhaul demands. Defaults to 0.2.
            backhaul_class(int): Class of backhaul problem. If 1, it's unmixed, if 2, it's mixed. Defaults to 1.
            sample_backhaul_class(bool): If backhaul class is sampled across batches. Defaults to False.
            max_distance_limit(float): Route distance limits. Defaults to 2.8.
            speed(float): Vehicles' speed. Defaults to 1.0.
            subsample(bool): If problem variants are to be sampled. Defaults to True.
            variant_preset(str): Variant preset to be sampled. Defaults to "all".
            use_combinations(bool): It considers combinations for which sampling mask the instance is defined. Defaults to False.
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int): Number of augmentations. Defaults to 2.
            sample_type(str): Sample type. It can be "random", "augment" or "saved". Defaults to "random".
            instance_name(str): Instance file path. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".

        Returns:
            TensorDict: Instance data.
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

        if instance_name==None and random_sample==False:
            instance_name = self.sample_name_from_set(seed=seed)
        elif instance_name==None and random_sample==True:
            instance_name = 'random_instance'
        else:
            instance_name = instance_name

        if num_depots is None:
            self.num_depots = 3
        else:
            self.num_depots = num_depots

        if num_agents is None:
            self.num_agents = 2
            self.num_agents_total = self.num_agents * self.num_depots
        else:
            self.num_agents = num_agents
            self.num_agents_total = self.num_agents * self.num_depots

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
            self.capacity = get_vehicle_capacity(self.num_nodes)
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

        if variant_preset is None:
            self.variant_preset = 'all'
        else:
            self.variant_preset = variant_preset

        self.subsample = subsample
        self.use_combinations = use_combinations
        self.seed = seed
        self.device = device
        self.n_augment = n_augment

        if variant_preset in VARIANT_PRESETS_MIXED:
            self.variant_preset = variant_preset.replace('m', '', 1)
            self.backhaul_class = 2

        if sample_type == 'random':
            instance_info = self.random_generate_instance(
                num_depots = self.num_depots,
                num_agents = self.num_agents,
                num_nodes = self.num_nodes,
                min_coords = self.min_coords,
                max_coords = self.max_coords,
                capacity = self.capacity,
                service_time = self.service_time,
                min_demands = self.min_demands,
                max_demands = self.max_demands,
                min_backhaul = self.min_backhaul,
                max_backhaul = self.max_backhaul,
                max_time = self.max_time,
                backhaul_ratio = self.backhaul_ratio,
                backhaul_class = self.backhaul_class,
                sample_backhaul_class = self.sample_backhaul_class,
                max_distance_limit = self.max_distance_limit,
                speed = self.speed,
                subsample = self.subsample,
                variant_preset = self.variant_preset,
                use_combinations = self.use_combinations,
                batch_size = self.batch_size,
                seed = self.seed,
                device = self.device
            )
            
        elif sample_type == 'augment':
            instance_info = self.augment_generate_instance(
                num_depots = self.num_depots,
                num_agents = self.num_agents,
                num_nodes = self.num_nodes,
                min_coords = self.min_coords,
                max_coords = self.max_coords,
                capacity = self.capacity,
                service_time = self.service_time,
                min_demands = self.min_demands,
                max_demands = self.max_demands,
                min_backhaul = self.min_backhaul,
                max_backhaul = self.max_backhaul,
                max_time = self.max_time,
                backhaul_ratio = self.backhaul_ratio,
                backhaul_class = self.backhaul_class,
                sample_backhaul_class = self.sample_backhaul_class,
                max_distance_limit = self.max_distance_limit,
                speed = self.speed,
                subsample = self.subsample,
                variant_preset = self.variant_preset,
                use_combinations = self.use_combinations,
                n_augment=self.n_augment,
                batch_size = self.batch_size,
                seed = self.seed,
                device = self.device
            )

        elif sample_type=='saved':
            instance_info = self.get_instance(instance_name, num_agents=num_agents)

        return instance_info
        

    
if __name__ == "__main__":

    MIXED_PROBLEMS = ["ovrpmb", "ovrpmbl", "ovrpmbltw", "ovrpmbtw",
                      "vrpmb", "vrpmbl", "vrpmbltw", "vrpmbtw"]
    

    number_instances = 2
    print("Starting validation/test sets generation...")
    print()

    for num_nodes, n_agent, num_depots in [(26, 3, 3)]:
        generator = InstanceGenerator(batch_size=32, seed=0)

        for problem in VARIANT_PRESETS:

            for k in range(number_instances):

                #If problem is mixed, sample instance with another preset and backhaul_class=2
                if problem not in MIXED_PROBLEMS:
                    instance =  generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset=problem)
                else:
                    if problem == "ovrpmb":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset="ovrpb", backhaul_class=2)
                    elif problem == "ovrpmbl":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset="ovrpbl", backhaul_class=2)
                    elif problem == "ovrpmbltw":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset="ovrpbltw", backhaul_class=2)
                    elif problem == "ovrpmbtw":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset="ovrpbtw", backhaul_class=2)
                    elif problem == "vrpmb":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset="vrpb", backhaul_class=2)
                    elif problem == "vrpmbl":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset="vrpbl", backhaul_class=2)
                    elif problem == "vrpmbltw":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset="vrpbltw", backhaul_class=2)
                    elif problem == "vrpmbtw":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset="vrpbtw", backhaul_class=2)
                    else:
                        raise Exception("Error generating validation set.")

                name = f'generated_val_servs_{num_nodes}_agents_{n_agent}_depots_{num_depots}_{problem}_{k}'
                instance['name'] = name
                print("Generating validation data...")
                if not os.path.exists(f'data/generated/servs_{num_nodes-num_depots}_agents_{n_agent}/{problem}/validation'):
                    os.makedirs(f'data/generated/servs_{num_nodes-num_depots}_agents_{n_agent}/{problem}/validation')
                    print(f"Creating directory: data/generated/servs_{num_nodes-num_depots}_agents_{n_agent}/{problem}/validation")
                with open(f'data/generated/servs_{num_nodes-num_depots}_agents_{n_agent}/{problem}/validation/'+name+'.pkl', 'wb') as fp:
                    pickle.dump(instance, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Dumped data into: data/generated/servs_{num_nodes-num_depots}_agents_{n_agent}/{problem}/validation/{name}.pkl")

                #If problem is mixed, sample instance with another preset and backhaul_class=2
                if problem not in MIXED_PROBLEMS:
                    instance =  generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset=problem)
                else:
                    if problem == "ovrpmb":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset="ovrpb", backhaul_class=2)
                    elif problem == "ovrpmbl":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset="ovrpbl", backhaul_class=2)
                    elif problem == "ovrpmbltw":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset="ovrpbltw", backhaul_class=2)
                    elif problem == "ovrpmbtw":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset="ovrpbtw", backhaul_class=2)
                    elif problem == "vrpmb":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset="vrpb", backhaul_class=2)
                    elif problem == "vrpmbl":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset="vrpbl", backhaul_class=2)
                    elif problem == "vrpmbltw":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset="vrpbltw", backhaul_class=2)
                    elif problem == "vrpmbtw":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, num_depots=num_depots, variant_preset="vrpbtw", backhaul_class=2)
                    else:
                        raise Exception("Error generating test set.")
                
                name = f'generated_test_servs_{num_nodes}_agents_{n_agent}_depots_{num_depots}_{problem}_{k}'
                instance['name'] = name
                print("Generating test data...")
                if not os.path.exists(f'data/generated/servs_{num_nodes-num_depots}_agents_{n_agent}/{problem}/test'):
                    os.makedirs(f'data/generated/servs_{num_nodes-num_depots}_agents_{n_agent}/{problem}/test')
                    print(f"Creating directory: data/generated/servs_{num_nodes-num_depots}_agents_{n_agent}/{problem}/test")
                with open(f'data/generated/servs_{num_nodes-num_depots}_agents_{n_agent}/{problem}/test/'+name+'.pkl', 'wb') as fp:
                    pickle.dump(instance, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Dumped data into: data/generated/servs_{num_nodes-num_depots}_agents_{n_agent}/{problem}/test/{name}.pkl")

    print('Generation completed.')