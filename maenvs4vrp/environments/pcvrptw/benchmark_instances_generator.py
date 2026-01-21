import torch
from tensordict import TensorDict

import os
from os import path

from typing import Dict, Optional
from maenvs4vrp.core.env_generator_builder import InstanceBuilder


BENCHMARK_INSTANCES_PATH = 'pcvrptw/data/benchmark'

class BenchmarkInstanceGenerator(InstanceBuilder):
    """
    PCVRPTW benchmark instance generation class.
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
        base_dir = path.dirname(path.dirname(path.abspath(__file__)))

        return {'Solomon': [s.split('.')[0] for s in os.listdir(path.join(base_dir, BENCHMARK_INSTANCES_PATH, 'Solomon'))],
                'Homberger':[s.split('.')[0] for s in os.listdir(path.join(base_dir, BENCHMARK_INSTANCES_PATH, 'Homberger'))]}

    def __init__(self, 
                 instance_type:str='Solomon', 
                 set_of_instances:set=None, 
                 device: Optional[str] = "cpu",
                 batch_size: Optional[torch.Size] = None,
                 seed:int=None) -> None:
        """
        Constructor. Create an instance space of one or several sets of data.
        
        Args:       
            instance_type(str): Instance type. Can be "Solomon" or "Homberger". Defaults to "Solomon".
            set_of_instances(set): Set of instances file names. Defaults to None.
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

        assert instance_type in ["Solomon", "Homberger"], f"instance unknown type"
        assert len(set_of_instances)>0, f"set_of_instances has to have at least one instance!"
        if set_of_instances:
            self.instance_type = instance_type
            self.set_of_instances = set_of_instances
            self.load_set_of_instances()


    def read_instance_data(self, instance_name:str)-> Dict:
        """
        Read instance data from file.

        Args:
            instance_name(str): Instance file name.

        Returns: 
            Dict: Instance data.
        """

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))

        path_to_file = path.join(base_dir, BENCHMARK_INSTANCES_PATH, self.instance_type)

        benchmark_file = '{path_to_benchmark_instances}/{instance}.txt' \
                        .format(path_to_benchmark_instances=path_to_file,
                                instance=instance_name)

        dfile = open(benchmark_file)

        data = [[x for x in line.split()] for line in dfile]
        dfile.close()
        instance = self.parse_instance_data(data)
        return instance

    def parse_instance_data(self, instance_data: list) -> Dict:
        """
        Parse instance data list into a dictionary.

        Args:
            instance_data(list): Instance data.

        Returns:
            Dict: Parsed instance data.
        """
        instance = dict()
        instance['name'] = instance_data[0][0]

        coords = []
        demands = []
        time_windows = []
        service_time = []

        for data in instance_data[9:]:
            coords.append([float(data[1]), float(data[2])])
            demands.append(float(data[3]))
            time_windows.append([float(data[4]), float(data[5])])
            service_time.append(float(data[6]))

        instance['num_agents'] = int(instance_data[4][0])
        instance['num_nodes'] = len(coords) # -1 to account for depot

        data = TensorDict({}, batch_size=self.batch_size, device=self.device)

        capacity = float(instance_data[4][1])

        depot_idx = 0
        data['depot_idx'] = depot_idx * torch.ones((*self.batch_size, 1), dtype = torch.int64, device=self.device)
        data['coords'] = torch.tensor(coords, dtype = torch.float, device=self.device).unsqueeze(0)

        data['demands'] = torch.tensor(demands, dtype = torch.float, device=self.device).unsqueeze(0)
        profits = torch.clip(torch.zeros_like(data['demands'], device=self.device).uniform_(0.75, 2.25)*data['demands'], min=1)
        profits[:, depot_idx] = 0.0
        data['profits'] = torch.round(profits)
        time_windows = torch.tensor(time_windows, dtype = torch.float, device=self.device).unsqueeze(0)
        data['tw_low'] =  time_windows[:, :, 0].clone()
        data['tw_high'] = time_windows[:, :, 1].clone()

        data['service_time'] = torch.tensor(service_time, dtype = torch.float, device=self.device).unsqueeze(0)
        data['start_time'] = time_windows[:, :, 0].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                          dtype=torch.int64, device=self.device)).squeeze(-1)
        data['end_time'] = time_windows[:, :, 1].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                        dtype=torch.int64, device=self.device)).squeeze(-1)


        data['is_depot'] = torch.zeros((*self.batch_size, instance['num_nodes']), dtype=torch.bool, device=self.device)
        data['is_depot'][:, depot_idx] = True
        data['capacity'] = capacity * torch.ones((*self.batch_size, 1), dtype = torch.float, device=self.device)

        instance['data'] = data

        if self.instance_type in ['Solomon', 'Homberger']:
            instance['n_digits'] = 10.0
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
            num_agents = min(instance['num_agents'], num_agents)
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


    def sample_first_n_services(self, 
                               instance_name:str=None,
                                num_agents:int=None, 
                                num_nodes:int=None,
                                device:Optional[str]="cpu")-> Dict:
        """
        Sample first n nodes. 

        Args:
            instance_name(str): Instance file name. Defaults to None.
            num_agents(int): Total number of agents. Defaults to None.
            num_nodes(int): Total number of (n) nodes intended. Defaults to None.

        Returns:
            Dict: New instance of the first n nodes.
        """

        new_instance = dict()
        instance = self.get_instance(instance_name, num_agents)

        new_instance['num_agents'] = instance['num_agents']

        if num_nodes is not None:
            num_nodes = min(num_nodes, instance['num_nodes'])
            new_instance['num_nodes'] = num_nodes
        else:
            new_instance['num_nodes'] = instance['num_nodes']

        new_instance['name'] = instance['name'] + '_samp'
        new_instance['n_digits'] = instance['n_digits'] 
        data = instance['data']

        idxs = torch.arange(0, instance['num_nodes'], device=self.device)
        index = idxs[:new_instance['num_nodes']]

        new_data = TensorDict({}, batch_size=self.batch_size, device=self.device)

        new_data['capacity'] = data['capacity']
        new_data['depot_idx'] = data['depot_idx']
        new_data['coords'] = data['coords'][:, index]
        new_data['demands'] = data['demands'][:,index]
        new_data['profits'] = data['profits'][:,index]
        new_data['tw_low'] = data['tw_low'][:,index]
        new_data['tw_high'] = data['tw_high'][:,index]
        new_data['service_time'] = data['service_time'][:,index]
        new_data['start_time'] = data['start_time']
        new_data['end_time'] = data['end_time']
        new_data['is_depot'] = data['is_depot'][:, index]

        new_instance['data'] = new_data

        return new_instance

    def random_sample_instance(self, 
                               instance_name:str=None,
                               num_agents:int=None, 
                               num_nodes:int=None, 
                               seed:int=None,
                               device:Optional[str]="cpu")-> Dict:
        """
        Sample one instance from instance space, randomly adjusting the nodes.

        Args:
            instance_name(str): Instance file name. Defaults to None.
            num_agents(int):  Total number of agents. Defaults to None.
            num_nodes(int):  Total number of nodes. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            Dict: Instance data.
        """
        if seed is not None:
            self._set_seed(seed)

        new_instance = dict()
        instance = self.get_instance(instance_name, num_agents)

        new_instance['num_agents'] = instance['num_agents']

        if num_nodes is not None:
            num_nodes = min(num_nodes, instance['num_nodes'])
            new_instance['num_nodes'] = num_nodes
        else:
            new_instance['num_nodes'] = instance['num_nodes']

        new_instance['name'] = instance['name'] + '_samp'

        new_instance['n_digits'] = instance['n_digits'] 

        data = instance['data']

        idxs = torch.arange(0, instance['num_nodes'], device=self.device).unsqueeze(0)
        index = torch.cat([idxs[data['is_depot']], idxs[~data['is_depot']][torch.randperm(instance['num_nodes']-1)]], dim=0)[:new_instance['num_nodes']]

        new_data = TensorDict({}, batch_size=self.batch_size, device=self.device)

        new_data['capacity'] = data['capacity']
        new_data['depot_idx'] = data['depot_idx']
        new_data['coords'] = data['coords'][:, index]
        new_data['demands'] = data['demands'][:,index]
        new_data['profits'] = data['profits'][:,index]
        new_data['tw_low'] = data['tw_low'][:,index]
        new_data['tw_high'] = data['tw_high'][:,index]
        new_data['service_time'] = data['service_time'][:,index]
        new_data['start_time'] = data['start_time']
        new_data['end_time'] = data['end_time']
        new_data['is_depot'] = data['is_depot'][:, index]

        new_instance['data'] = new_data
        return new_instance

    def sample_name_from_set(self, seed:int=None)-> str:
        """
        Sample one instance from instance set.

        Args:
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            str: Instance sample name.
        """
        if seed is not None:
            self._set_seed(seed)
        assert len(self.set_of_instances)>0, f"set_of_instances has to have at least one instance!"

        return list(self.set_of_instances)[torch.randint(0, len(self.set_of_instances), (1,)).item()]

    def sample_instance(self, num_agents:int=None, 
                        num_nodes:int=None, 
                        capacity:int=None, 
                        service_times:float=None, 
                        profits:str=None,
                        instance_name:str=None, 
                        sample_type:str='random',
                        batch_size: Optional[torch.Size] = None,
                        n_augment: Optional[int] = None,
                        seed:int=None,
                        device:Optional[str]="cpu")-> Dict:
        """
        Sample one instance from instance space.

        Args:
            num_agents(int): Total number of agents. Defaults to None.
            num_nodes(int): Total number of nodes. Defaults to None.
            capacity(int): Capacity of the agents. Defaults to None.
            service_times(float): Service time in the nodes. Defaults to None.
            instance_name(str): Instance name. Defaults to None.
            sample_type(str): Sample type. It can be "random" or something else for "first n". Defaults to "random".
            batch_size(torch.Size or None): Batch size. Defaults to None.
            n_augment(int, optional): Data augmentation. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            Dict: Instance data.
        """
        if seed is not None:
            self._set_seed(seed)

        if instance_name==None:
            instance_name = self.sample_name_from_set(seed=seed)
        else:
            instance_name = instance_name

        if sample_type=='random':
            instance = self.random_sample_instance(instance_name=instance_name,
                                                   num_agents=num_agents, 
                                                   num_nodes=num_nodes, 
                                                   seed=seed,
                                                   device=device)
        else:
            # sample first n
            instance = self.sample_first_n_services(instance_name=instance_name,
                                                    num_agents=num_agents, 
                                                    num_nodes=num_nodes,
                                                    device=device)

        return instance


if __name__ == '__main__':

    generator = BenchmarkInstanceGenerator(instance_type='Solomon', set_of_instances={'C101'})
    generator.sample_instance(num_agents=3, num_nodes=8, seed=1)