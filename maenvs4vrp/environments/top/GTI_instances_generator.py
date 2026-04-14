import torch
from tensordict import TensorDict

import os
from os import path
import pickle

from typing import Dict, Optional
from maenvs4vrp.core.env_generator_builder import InstanceBuilder

GENERATED_INSTANCES_PATH = 'top/data/benchmark'

class GTIGenerator(InstanceBuilder):
    """
    class for TOP GTI benchmark instances generation
    
    """
    @classmethod
    def get_list_of_benchmark_instances(cls):
        base_dir = path.dirname(path.dirname(path.abspath(__file__)))

        return {'GTI_20': [s.split('.')[0] for s in os.listdir(path.join(base_dir, GENERATED_INSTANCES_PATH, 'GTI')) if '20_L2_' in s],
                'GTI_50': [s.split('.')[0] for s in os.listdir(path.join(base_dir, GENERATED_INSTANCES_PATH, 'GTI')) if '50_L2_' in s],
                'GTI_100': [s.split('.')[0] for s in os.listdir(path.join(base_dir, GENERATED_INSTANCES_PATH, 'GTI')) if '100_L2_' in s],}
        
    def __init__(self, 
                 instance_type:str='GTI', 
                 set_of_instances:set=None, 
                 device: Optional[str] = "cpu",
                 batch_size: Optional[torch.Size] = None,
                 seed:int=None) -> None:
        """    

        Args:       
            instance_type (str):  instance type. Can be "validation" or "test";
            set_of_instances (bool):  set of instances file names;
            seed (int): random number generator seed. Defaults to None;
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

        assert instance_type in ["GTI"], f"instance unknown type"
        self.set_of_instances = set_of_instances
        if set_of_instances:
            self.instance_type = instance_type
            self.load_set_of_instances()
            

    def read_instance_data(self, instance_name:str)-> Dict:
        """
        Reads instance data
        Args:
            instance_name (str): instance file name.

        Returns: 
            Dict: Instance data 
        """

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
        path_to_file = path.join(base_dir, GENERATED_INSTANCES_PATH, self.instance_type)
        generated_file = '{path_to_generated_instances}/{instance}.pkl' \
                        .format(path_to_generated_instances=path_to_file,
                                instance=instance_name)
        with open(generated_file, 'rb') as fp:
            data = pickle.load(fp)
            data = [
                        {
                            'loc': torch.FloatTensor(loc),
                            'prize': torch.FloatTensor(prize),
                            'depot': torch.FloatTensor(depot),
                            'max_length': torch.tensor(length)
                        }
                        for depot, loc, prize, length in data
                    ]
        instance = self.parse_instance_data(data, instance_name)

        return instance


    def get_instance(self, instance_name:str, num_agents:int=None) -> Dict:
        """
        Returns:
            Dict: Instance data

        """
        instance = self.instances_data.get(instance_name)

        if num_agents is not None:
            assert num_agents>0, f"number of agents must be grater them 0!"
            instance['num_agents'] = num_agents

        return instance

    def parse_instance_data(self, instance_data: list, instance_name:str) -> Dict:
        """
        Parse instance data into dict
        
        """
        instance = dict()
        instance['name'] = instance_name

        coords = []
        profits = []
        max_length = []
        self.service_times = 0.0
        for data in instance_data:
            coords.append(torch.cat((data['depot'][None, :], data['loc']), -2).unsqueeze(0))
            profits.append(torch.cat((torch.tensor([0]), data['prize']), -1).unsqueeze(0))
            max_length.append(data['max_length'])

        self.batch_size = torch.Size([len(instance_data)])
        data = TensorDict({}, batch_size=self.batch_size, device=self.device)

        self.depot_idx = 0
        data['depot_idx'] = self.depot_idx * torch.ones((*self.batch_size, 1), dtype = torch.int64, device=self.device)
        data['coords'] = torch.cat(coords, dim=0).to(self.device)

        num_nodes = data['coords'].shape[1]
        instance['num_agents'] = 4 
        instance['num_nodes'] = num_nodes 

        data['profits'] = torch.cat(profits, dim=0).to(self.device)

        service_times = self.service_times * torch.ones((*self.batch_size, num_nodes), dtype = torch.float, device=self.device)
        service_times[:, self.depot_idx] = 0
        data['service_time'] = service_times
        data['speed'] = torch.ones((*self.batch_size, 1), dtype=torch.float, device=self.device)

        data['start_time'] = torch.zeros((*self.batch_size, 1), device=self.device).squeeze(-1)

        data['end_time'] = torch.tensor(max_length, device=self.device)


        data['is_depot'] = torch.zeros((*self.batch_size, instance['num_nodes']), dtype=torch.bool, device=self.device)
        data['is_depot'][:, self.depot_idx] = True

        instance['data'] = data

        return instance


    def load_set_of_instances(self, set_of_instances:set=None):
        """
        Loads every instance on set_of_instances set
        
        Args:
            set_of_instances (set): set of instances file names. Defaults to None.

        """
        if set_of_instances:
            self.set_of_instances = set_of_instances
        self.instances_data = dict()
        for instance_name in self.set_of_instances:
            instance = self.read_instance_data(instance_name)
            self.instances_data[instance_name] = instance

    
    def sample_name_from_set(self, seed:int=None)-> str:
        """
        Samples one instance from instance set

        Args:
            seed (int): random number generator seed. Defaults to None;

        Returns:
            str: instance name.
        """
        if seed is not None:
            self._set_seed(seed)
        assert len(self.set_of_instances)>0, f"set_of_instances has to have at least one instance!"

        return list(self.set_of_instances)[torch.randint(0, len(self.set_of_instances), (1,)).item()]

    def sample_instance(self, 
                        num_agents=None, 
                        num_nodes=None, 
                        service_times=0.0,
                        speed:float=None,
                        profits:str='constant',
                        instance_name:str=None, 
                        sample_type:str='saved',
                        batch_size: Optional[torch.Size] = None,
                        n_augment: Optional[int] = None,
                        seed:int=None)-> Dict:
        """
        Samples one instance from instance space

        Args:
            num_agents (int): Total number of agents. Defaults to 20.
            num_nodes (int):  Total number of nodes. Defaults to 100.
            service_times (int): Total time of service. Defaults to 0.2.            
            speed (float): Vehicles' speed. Defaults to None.
            instance_name (str):  instance name. Defaults to None;
            random_sample (bool):  True to sample instance and False to use original instance data. Defaults to None;
            seed (int): random number generator seed. Defaults to None;

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
        if num_nodes is None:
            num_nodes = 20
        if service_times is None:
            service_times = 0.0
        if speed is None:
            self.speed = 1.0
        else:
            self.speed = speed

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)
           
        if sample_type in ['random', 'augment']:
            raise NotImplementedError()         
        elif sample_type=='saved':
            instance_info = self.get_instance(instance_name, num_agents=num_agents)

        return instance_info

if __name__ == '__main__':

    print('done')
