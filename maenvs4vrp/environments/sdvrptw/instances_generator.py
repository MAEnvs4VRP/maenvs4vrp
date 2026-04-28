import torch
from tensordict import TensorDict

import os
from os import path
import pickle

from typing import Dict, Optional
from maenvs4vrp.core.env_generator_builder import InstanceBuilder

GENERATED_INSTANCES_PATH = 'sdvrptw/data/generated'

class InstanceGenerator(InstanceBuilder):
    """
    SDVRPTW instance generation class.
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
            val_path = path.join( GENERATED_INSTANCES_PATH, folder, 'validation')
            test_path = path.join(GENERATED_INSTANCES_PATH, folder, 'test')
            benchmark_instances[folder] = {'validation': [val_path + '/' + s.split('.')[0] for s in os.listdir(path.join(base_dir, val_path))],
                                            'test':[test_path + '/' + s.split('.')[0] for s in os.listdir(path.join(base_dir, test_path))]}
        return benchmark_instances

    def __init__(self,
                 instance_type:str='validation',
                 set_of_instances:set=None,
                 device: Optional[str] = "cpu",
                 batch_size: Optional[torch.Size] = None,
                 seed:int=None) -> None:
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

        self.max_num_agents = 20
        self.max_num_nodes = 100

        assert instance_type in ["validation", "test"], f"instance unknown type"
        self.set_of_instances = set_of_instances
        if set_of_instances:
            self.instance_type = instance_type
            self.load_set_of_instances()


    def read_instance_data(self, instance_name:str)-> Dict:
        """
        Read instance data from file.

        Args:
            instance_name(str): instance file name.

        Returns:
            Dict: Instance data.
        """

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
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

    def load_set_of_instances(self, set_of_instances:set=None):
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


    def get_time_windows(self,
                         instance:TensorDict=None,
                         batch_size:torch.Size=None,
                         seed:int=None)-> torch.tensor:
        """
        Get time windows to reach the nodes.

        Args:
            instance(TensorDict): Data instance. Defaults to None.
            batch_size(torch.Size): Batch size. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            torch.Tensor: Nodes time windows.
        """

        if seed is not None:
            self._set_seed(seed)

        time_windows = torch.zeros((*batch_size, self.max_num_nodes, 2), device=self.device)

        depot_coord = instance['coords'].gather(1, instance['depot_idx'][:, :, None].expand(-1, -1, 2))
        dist_depot = torch.pairwise_distance(depot_coord, instance['coords'], keepdim = True)

        depot_start, depot_end = 0, 3

        inf = depot_start + dist_depot
        sup = depot_end - dist_depot - self.service_times

        time_centers = inf.squeeze(-1) + torch.rand(*batch_size, self.max_num_nodes, device=self.device) * (sup-inf).squeeze(-1)
        time_half_width = torch.empty((*batch_size, self.max_num_nodes), device=self.device).uniform_(self.service_times / 2 , depot_end / 3)
        time_windows[:, :, 0] = torch.clip(time_centers - time_half_width, depot_start, depot_end)
        time_windows[:, :, 1] = torch.clip(time_centers + time_half_width, depot_start, depot_end)
        time_windows[:, self.depot_idx, 0] = depot_start
        time_windows[:, self.depot_idx, 1] = depot_end

        return time_windows


    def random_generate_instance(self, num_agents:int=20,
                                 num_nodes:int=100,
                                 capacity:int=50,
                                 service_times:int=0.2,
                                 speed:float=None,
                                 batch_size: Optional[torch.Size] = None,
                                 seed:int=None,
                                 device:Optional[str]="cpu")-> TensorDict:
        """
        Generate random instance.

        Args:
            num_agents(int): Total number of agents. Defaults to 20.
            num_nodes(int):  Total number of nodes. Defaults to 100.
            capacity(int): Total capacity for each agent. Defaults to 50.
            service_times(int): Total time of service. Defaults to 0.2.
            speed(float): Vehicles' speed. Defaults to None.
            batch_size(torch.Size, optional): Batch size. Defaults to None.
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
        if service_times is not None:
            self.service_times = service_times
        if capacity is not None:
            assert capacity>0, f"agent capacity must be grater them 0!"
            self.capacity = capacity
        if speed is not None:
            assert speed>0, f"Speed must be greater than 0!"

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)

        self.depot_idx = 0
        instance['depot_idx'] = self.depot_idx * torch.ones((*self.batch_size, 1), dtype = torch.int64, device=self.device)

        coords = torch.rand(*self.batch_size, self.max_num_nodes, 2, dtype = torch.float, device=self.device)
        instance['coords'] = coords

        demands = torch.randint(low = 1, high=11, size = (*self.batch_size, num_nodes), dtype = torch.float, device=self.device)
        demands[:, self.depot_idx] = 0.0

        instance['demands'] = demands
        service_times = self.service_times * torch.ones((*self.batch_size, num_nodes), dtype = torch.float, device=self.device)
        service_times[:, self.depot_idx] = 0
        instance['service_time'] = service_times
        instance['speed'] = torch.full((*self.batch_size, 1), self.speed, dtype=torch.float32)

        time_windows = self.get_time_windows(instance, self.batch_size, seed)

        instance['tw_low'] =  time_windows[:, :, 0].clone()
        instance['tw_high'] = time_windows[:, :, 1].clone()

        instance['is_depot'] = torch.zeros((*self.batch_size, num_nodes), dtype=torch.bool, device=self.device)
        instance['is_depot'][:, self.depot_idx] = True

        instance['start_time'] = time_windows[:, :, 0].gather(1, torch.zeros((*self.batch_size, 1),
                                                                          dtype=torch.int64, device=self.device)).squeeze(-1)
        instance['end_time'] = time_windows[:, :, 1].gather(1, torch.zeros((*self.batch_size, 1),
                                                                        dtype=torch.int64, device=self.device)).squeeze(-1)
        instance['capacity'] = self.capacity * torch.ones((*self.batch_size, 1), dtype = torch.float, device=self.device)

        instance_info = {'name':'random_instance',
                         'num_nodes': self.max_num_nodes,
                         'num_agents':self.max_num_agents,
                         'data':instance}
        return instance_info

    def augment_generate_instance(self, num_agents:int=20,
                                 num_nodes:int=100,
                                 capacity:int=50,
                                 service_times:int=0.2,
                                 speed:float=None,
                                 batch_size: Optional[torch.Size] = None,
                                 n_augment:int = 2,
                                 seed:int=None,
                                 device:Optional[str]="cpu")-> TensorDict:
        """
        Generate augmentated instance.

        Args:
            num_agents(int): Total number of agents. Defaults to 20.
            num_nodes(int):  Total number of nodes. Defaults to 100.
            capacity(int): Total capacity for each agent. Defaults to 50.
            service_times(int): Service time in the nodes. Defaults to 0.2.
            speed(float): Vehicles' speed. Defaults to None.
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int): Data augmentation. Defaults to 2.
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
        if service_times is not None:
            self.service_times = service_times
        if capacity is not None:
            assert capacity>0, f"agent capacity must be grater them 0!"
            self.capacity = capacity
        if speed is not None:
            assert speed>0, f"Speed must be greater than 0!"

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        assert self.batch_size.numel()%n_augment == 0, f"batch_size must be divisible by n_augment"
        s_batch_size = self.batch_size.numel() // n_augment
        self.s_batch_size = torch.Size([s_batch_size])

        instance_info_s = self.random_generate_instance(num_agents=num_agents,
                                                     num_nodes=num_nodes,
                                                     capacity=capacity,
                                                     service_times=service_times,
                                                     speed=speed,
                                                     batch_size = self.s_batch_size,
                                                     seed=seed,
                                                     device=device)

        self.batch_size = torch.Size(batch_size)

        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)
        for key in instance_info_s['data'].keys():
            if len(instance_info_s['data'][key].shape) == 3:
                instance[key] = instance_info_s['data'][key].repeat(n_augment, 1, 1)
            elif len(instance_info_s['data'][key].shape) == 2:
                instance[key] = instance_info_s['data'][key].repeat(n_augment, 1)
            elif len(instance_info_s['data'][key].shape) == 1:
                instance[key] = instance_info_s['data'][key].repeat(n_augment)

        instance_info = {'name':'random_instance',
                         'num_nodes': self.max_num_nodes,
                         'num_agents':self.max_num_agents,
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

    def sample_instance(self,
                        num_agents=None,
                        num_nodes=None,
                        capacity=50,
                        service_times=0.2,
                        speed:float=1.0,
                        instance_name:str=None,
                        sample_type:str='random',
                        batch_size: Optional[torch.Size] = None,
                        n_augment: Optional[int] = None,
                        seed:int=None,
                        device: Optional[str] = "cpu")-> Dict:
        """
        Sample one instance from instance space.

        Args:
            num_agents(int): Total number of agents. Defaults to None.
            num_nodes(int):  Total number of nodes. Defaults to None.
            capacity(int): Total capacity for each agent. Defaults to 50.
            service_times(int): Service time in the nodes. Defaults to 0.2.
            speed(float): Vehicles' speed. Defaults to 1.0.
            instance_name(str):  Instance name. Defaults to None.
            sample_type(str): Sample type. It can be "random", "augment" or "saved". Defaults to "random".
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
            num_agents = 20
        if num_nodes is None:
            num_nodes = 100
        if capacity is None:
            capacity = 50
        if service_times is None:
            service_times = 0.2
        if speed is None:
            self.speed = 1.0
        else:
            self.speed = speed

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        if sample_type=='random':
            instance_info = self.random_generate_instance(num_agents=num_agents,
                                                     num_nodes=num_nodes,
                                                     capacity=capacity,
                                                     service_times=service_times,
                                                     speed=self.speed,
                                                     batch_size = batch_size,
                                                     seed=seed,
                                                     device=device)
        elif sample_type=='augment':
            instance_info = self.augment_generate_instance(num_agents=num_agents,
                                                     num_nodes=num_nodes,
                                                     capacity=capacity,
                                                     service_times=service_times,
                                                     speed=self.speed,
                                                     batch_size = batch_size,
                                                     n_augment = n_augment,
                                                     seed=seed,
                                                     device=device)
        elif sample_type=='saved':
            instance_info = self.get_instance(instance_name, num_agents=num_agents)

        return instance_info

if __name__ == '__main__':

    number_instances = 64
    print('starting valid/test sets generation')

    # valid/test sets generation
    for num_nodes, n_agent in [(101, 25), (51, 25)]:
        generator = InstanceGenerator(batch_size=32, seed=0)
        for k in range(number_instances):
            instance =  generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes)
            name = f'generated_val_servs_{num_nodes-1}_agents_{n_agent}_{k}'
            instance['name'] = name
            if not os.path.exists(f'data/generated/servs_{num_nodes-1}_agents_{n_agent}/test'):
                os.makedirs(f'data/generated/servs_{num_nodes-1}_agents_{n_agent}/validation')
            with open(f'data/generated/servs_{num_nodes-1}_agents_{n_agent}/validation/'+name+'.pkl', 'wb') as fp:
                pickle.dump(instance, fp, protocol=pickle.HIGHEST_PROTOCOL)

            instance =  generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes)
            name = f'generated_test_servs_{num_nodes-1}_agents_{n_agent}_{k}'
            instance['name'] = name
            if not os.path.exists(f'data/generated/servs_{num_nodes-1}_agents_{n_agent}/test'):
                os.makedirs(f'data/generated/servs_{num_nodes-1}_agents_{n_agent}/test')
            with open(f'data/generated/servs_{num_nodes-1}_agents_{n_agent}/test/'+name+'.pkl', 'wb') as fp:
                pickle.dump(instance, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print('done')
