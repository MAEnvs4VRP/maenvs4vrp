from typing import Dict, Optional

import torch


class InstanceBuilder(object):
    """
    Instance generator base class.
    """

    DEFAULT_SEED = 2925

    def __init__(
        self,
        instance_type: str = None,
        set_of_instances: set = None,
        num_services: int = None,
        num_agents: int = None,
        seed: int = None,
        device: str = "cpu",
        batch_size: torch.Size = None,
    ) -> None:
        """
        Constructor

        Args:
            instance_type(str): Instance type. Defaults to none.
            set_of_instances(set):  Set of instances file names. Defaults to None.
            num_services(int):  Total number of services. Defaults to None.
            num_agents(int):  Total number of agents. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.
            device (str): Type of processing. Defaults to "cpu".
            batch_size(torch.Size or None): Batch size. If not specified, defaults to 1.
        """

        self.max_n_services = None
        self.max_n_vehicles = None
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

    def _set_seed(self, seed: Optional[int]):
        """
        Set the random seed used by the environment.

        Args:
            seed(int, optional): Seed used.

        Returns:
            None.
        """
        self.seed = seed
        rng = torch.manual_seed(self.seed)
        self.rng = rng

    def read_instance_data(self, instance_name: str) -> Dict:
        """
        Read instance data from file.

        Args:
            instance_name(str): instance file name.

        Returns:
            Dict: Instance data.
        """
        raise NotImplementedError()

    def get_instance(self, instance_name: str, preloaded: bool = False) -> Dict:
        """
        Combine read instance file and parse to Dict.

        Args:
            instance_name(str): Instance file name.
            preloaded(bool): If instance data has been pre-loaded. Defaults to False.

        Returns:
            Dict: Instance data.
        """
        raise NotImplementedError()

    def load_set_of_instances(
        self, set_of_instances: set = None, already_loaded: bool = None
    ):
        """
        Load every instance on set_of_instances set.

        Args:
            set_of_instances(set): Set of instances file names. Defaults to None.
            already_loaded(bool): If instance data has been pre-loaded. Defaults to None.

        """
        raise NotImplementedError()

    def get_instance_preloaded(self) -> Dict:
        """
        Get preloaded instance.

        Args:
            n/a.

        Returns:
            Dict: Instance data.
        """
        raise NotImplementedError()

    def random_sample_instance(
        self, num_agents: int = None, num_services: int = None, seed: int = None
    ) -> Dict:
        """
        Sample one instance from instance space.

        Args:
            num_services(int):  Total number of services. Defaults to None.
            num_agents(int):  Total number of agents. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            Dict: Instance data.
        """
        raise NotImplementedError()

    def sample_name_from_set(self, seed: int = None) -> str:
        """
        Sample one instance from insance set.

        Args:
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            str: instance name.
        """
        raise NotImplementedError()

    def sample_instance(
        self,
        num_agents: int = None,
        num_services: int = None,
        instance_name: str = None,
        random_sample: bool = True,
        seed: int = None,
    ) -> Dict:
        """
        Sample one instance from insance space.

        Args:
            num_services(int): Total number of services. Defaults to None.
            num_agents(int): Total number of agents. Defaults to None.
            instance_name(str): Instance name. Defaults to None.
            random_sample(bool): True to sample instance and False to use original instance data. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            Dict: Instance data.
        """
        raise NotImplementedError()
