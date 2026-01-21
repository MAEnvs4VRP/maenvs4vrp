import torch
from tensordict import TensorDict

import os
from os import path
import pickle

from typing import Dict, Optional
from maenvs4vrp.core.env_generator_builder import InstanceBuilder

GENERATED_INSTANCES_PATH = 'dsvrptw/data/generated'

class InstanceGenerator(InstanceBuilder):
    """
    DSVRPTW instance generation class.
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



    def random_generate_instance(self, num_agents:int=25, 
                                 num_nodes:int=100, 
                                 capacity:int=200, 
                                 min_cust_count:Optional[int] = None,
                                 cust_loc_range:tuple = (0,101),
                                 cust_dem_range:tuple= (5,41),
                                 horizon:int = 480,
                                 service_times_range:tuple = (10,31),
                                 tw_ratio:float = 0.5,
                                 cust_tw_range:tuple = (30,91),
                                 dod:float = 0.5,
                                 d_early_ratio:float= 0.5,
                                 batch_size: Optional[torch.Size] = None,
                                 seed:int=None,
                                 device:Optional[str]="cpu")-> TensorDict:
        """
        Generate random instance.

        Follows https://gitlab.inria.fr/gbono/mardam/-/blob/master/problems/_data_sdtw.py

        Args:
            num_agents (int): Total number of agents. Defaults to 25.
            num_nodes (int): Total number of nodes. Defaults to 100.
            capacity (int): Total capacity for each agent. Defaults to 200.
            min_cust_count (Optional[int]): Minimum customer count. Defaults to None.
            cust_loc_range (tuple): Range for customer locations. Defaults to (0, 101).
            cust_dem_range (tuple): Range for customer demands. Defaults to (5, 41).
            horizon (int): Planning horizon. Defaults to 480.
            service_times_range (tuple): Range for service times. Defaults to (10, 31).
            tw_ratio (float): Ratio of nodes with time windows. Defaults to 0.5.
            cust_tw_range (tuple): Range for customer time window widths. Defaults to (30, 91).
            dod (float): Ratio of dynamic nodes. Defaults to 0.5.
            d_early_ratio (float): Ratio of early dynamic nodes. Defaults to 0.5.
            batch_size (Optional[torch.Size]): Batch size. Defaults to None.
            seed (int, optional): Random number generator seed. Defaults to None.

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
        if service_times_range is not None:
            self.service_times_range = service_times_range
        if capacity is not None:
            assert capacity>0, f"agent capacity must be grater them 0!"
            self.capacity = capacity

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        size = (*self.batch_size, num_nodes)

        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)
        
        self.depot_idx = 0
        instance['depot_idx'] = self.depot_idx * torch.ones((*self.batch_size, 1), dtype = torch.int64, device=self.device)

        # Sample coords        x_j, y_j ~ U(0, 100)
        locs = torch.randint(*cust_loc_range, (*self.batch_size, num_nodes, 2), dtype = torch.float)
        # Sample demands             q_j ~ U(5,  40)
        demands = torch.randint(*cust_dem_range, (*self.batch_size, num_nodes), dtype = torch.float)
        demands[:, self.depot_idx] = 0.0

        # Sample service_times       s_j ~ U(10, 30)
        service_times = torch.randint(*service_times_range, (*self.batch_size, num_nodes), dtype = torch.float)
        service_times[:, self.depot_idx] = 0

        instance['coords'] = locs
        instance['demands'] = demands
        instance['service_time'] = service_times

        # Sample dyn subset           ~ B(dod)
        # and early/late appearance   ~ B(d_early_ratio)
        if isinstance(dod, float):
            is_dyn = torch.empty(size).bernoulli_(dod)
        elif len(dod) == 1:
            is_dyn = torch.empty(size).bernoulli_(dod[0])
        else: # tuple of float
            ratio = torch.tensor(dod)[torch.randint(0, len(dod), (batch_size,), dtype = torch.int64)]
            is_dyn = ratio[:,None,None].expand(*size).bernoulli()
        is_dyn[:, self.depot_idx] = 0

        if isinstance(d_early_ratio, float):
            is_dyn_e = torch.empty(size).bernoulli_(d_early_ratio)
        elif len(d_early_ratio) == 1:
            is_dyn_e = torch.empty(size).bernoulli_(d_early_ratio[0])
        else:
            ratio = torch.tensor(d_early_ratio)[
                    torch.randint(0, len(d_early_ratio), (batch_size,), dtype = torch.int64)
                    ]
            is_dyn_e = ratio[:,None,None].expand(*size).bernoulli()
        is_dyn_e[:, self.depot_idx] = 0

        # Sample appear. time     a_j = 0 if not in D subset
        #                         a_j ~ U(1,H/3) if early appear
        #                         a_j ~ U(H/3+1, 2H/3) if late appear
        aprs = is_dyn * is_dyn_e * torch.randint(1, horizon//3+1, size, dtype = torch.float) \
                + is_dyn * (1-is_dyn_e) * torch.randint(horizon//3+1, 2*horizon//3+1, size, dtype = torch.float)
        aprs[:, self.depot_idx] = 0


        # Sample TW subset            ~ B(tw_ratio)
        if isinstance(tw_ratio, float):
            has_tw = torch.empty(size).bernoulli_(tw_ratio)
        elif len(tw_ratio) == 1:
            has_tw = torch.empty(size).bernoulli_(tw_ratio[0])
        else: # tuple of float
            ratio = torch.tensor(tw_ratio)[torch.randint(0, len(tw_ratio), (batch_size,), dtype = torch.int64)]
            has_tw = ratio[:,None,None].expand(*size).bernoulli()
        
        # Sample TW width        tw_j = H if not in TW subset
        #                        tw_j ~ U(30,90) if in TW subset
        tws = (1 - has_tw) * torch.full(size, horizon) \
                + has_tw * torch.randint(*cust_tw_range, size, dtype = torch.float)
        tws[:, self.depot_idx] = horizon

        # Compute depot-to-customer travel times
        depot_locs = locs[:, self.depot_idx, :]  # shape: [batch_size, 2]
        cust_locs = locs  # shape: [batch_size, num_nodes, 2]
        tt_0j = (depot_locs.unsqueeze(1) - cust_locs).pow(2).sum(-1).pow(0.5)  # [batch_size, num_nodes]

        # Sample ready time
        rdys = has_tw * (aprs + torch.rand(size) * (horizon - torch.max(tt_0j + service_times, tws) - aprs))
        rdys.floor_()

        instance['tw_low'] =  rdys
        instance['tw_high'] = rdys + tws

        instance['is_depot'] = torch.zeros((*self.batch_size, num_nodes), dtype=torch.bool, device=self.device)
        instance['is_depot'][:, self.depot_idx] = True

        instance['start_time'] = torch.zeros(*self.batch_size, dtype=torch.int64, device=self.device)
        instance['end_time'] = horizon * torch.ones(*self.batch_size, dtype=torch.int64, device=self.device)
        instance['capacity'] = self.capacity * torch.ones((*self.batch_size, 1), dtype = torch.float, device=self.device)
        instance['appear_time'] =  aprs

        instance_info = {'name':'random_instance',
                         'late_penalty' : 2,
                         'num_nodes': self.max_num_nodes,
                         'num_agents':self.max_num_agents,
                         'data':instance}
        return instance_info

    def augment_generate_instance(self, num_agents:int=25, 
                                 num_nodes:int=100, 
                                 capacity:int=200, 
                                 min_cust_count:Optional[int] = None,
                                 cust_loc_range:tuple = (0,101),
                                 cust_dem_range:tuple= (5,41),
                                 horizon:int = 480,
                                 service_times_range:tuple = (10,31),
                                 tw_ratio:float = 0.5,
                                 cust_tw_range:tuple = (30,91),
                                 dod:float = 0.5,
                                 d_early_ratio:float= 0.5,
                                 batch_size: Optional[torch.Size] = None,
                                 n_augment:int = 2,
                                 seed:int=None,
                                 device:Optional[str]="cpu")-> TensorDict:
        """
        Generate augmentated instance.

        Args:
        Args:
            num_agents (int): Total number of agents. Defaults to 25.
            num_nodes (int): Total number of nodes. Defaults to 100.
            capacity (int): Total capacity for each agent. Defaults to 200.
            min_cust_count (Optional[int]): Minimum customer count. Defaults to None.
            cust_loc_range (tuple): Range for customer locations. Defaults to (0, 101).
            cust_dem_range (tuple): Range for customer demands. Defaults to (5, 41).
            horizon (int): Planning horizon. Defaults to 480.
            service_times_range (tuple): Range for service times. Defaults to (10, 31).
            tw_ratio (float): Ratio of nodes with time windows. Defaults to 0.5.
            cust_tw_range (tuple): Range for customer time window widths. Defaults to (30, 91).
            dod (float): Ratio of dynamic nodes. Defaults to 0.5.
            d_early_ratio (float): Ratio of early dynamic nodes. Defaults to 0.5.
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
        if service_times_range is not None:
            self.service_times_range = service_times_range
        if capacity is not None:
            assert capacity>0, f"agent capacity must be grater them 0!"
            self.capacity = capacity

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        assert self.batch_size.numel()%n_augment == 0, f"batch_size must be divisible by n_augment"
        s_batch_size = self.batch_size.numel() // n_augment
        self.s_batch_size = torch.Size([s_batch_size])
        
        instance_info_s = self.random_generate_instance(num_agents=num_agents, 
                                                     num_nodes=num_nodes, 
                                                     capacity=capacity, 
                                                     min_cust_count = min_cust_count,
                                                     cust_loc_range = cust_loc_range,
                                                     cust_dem_range= cust_dem_range,
                                                     horizon = horizon,
                                                     service_times_range = service_times_range,
                                                     tw_ratio= tw_ratio,
                                                     cust_tw_range = cust_tw_range,
                                                     dod = dod,
                                                     d_early_ratio = d_early_ratio, 
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
                         'late_penalty' : 2,
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


        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)
        else:
            batch_size = self.batch_size
            
        if sample_type=='random':
            instance_info = self.random_generate_instance(num_agents=num_agents, 
                                                     num_nodes=num_nodes, 
                                                     batch_size = batch_size,
                                                     seed=seed,
                                                     device=device)
        elif sample_type=='augment':
            instance_info = self.augment_generate_instance(num_agents=num_agents, 
                                                     num_nodes=num_nodes, 
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
            if not os.path.exists(f'data/generated/servs_{num_nodes-1}_agents_{n_agent}/validation'):
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
