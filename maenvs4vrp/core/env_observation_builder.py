from typing import Dict, List
import torch
from tensordict import TensorDict


class ObservationBuilder:
    """Observations base class."""

    POSSIBLE_NODES_STATIC_FEATURES: List[str] = []
    POSSIBLE_NODES_DYNAMIC_FEATURES: List[str] = []
    POSSIBLE_AGENT_FEATURES: List[str] = []
    POSSIBLE_OTHER_AGENTS_FEATURES: List[str] = []
    POSSIBLE_ALL_AGENTS_FEATURES: List[str] = []
    POSSIBLE_GLOBAL_FEATURES: List[str] = []

    def __init__(self, feature_list: Dict = None):
        """
        Constructor

        Args:
            feature_list(Dict): Dictionary containing observation features list to be available to the agent. Defaults to None.

        """
        self.default_feature_list = {
            "nodes_static": {},
            "nodes_dynamic": [],
            "agent": [],
            "other_agents": [],
            "all_agents": [],
            "global": [],
        }

        if feature_list is None:
            feature_list = self.default_feature_list

        self.feature_list = feature_list
        self.possible_nodes_static_features = self.POSSIBLE_NODES_STATIC_FEATURES
        self.possible_nodes_dynamic_features = self.POSSIBLE_NODES_DYNAMIC_FEATURES
        self.possible_agent_features = self.POSSIBLE_AGENT_FEATURES
        self.possible_agents_features = self.POSSIBLE_OTHER_AGENTS_FEATURES
        self.possible_all_agents_features = self.POSSIBLE_ALL_AGENTS_FEATURES
        self.possible_global_features = self.POSSIBLE_GLOBAL_FEATURES

    def set_env(self, env):
        """
        Set environment.

        Args:
            env(AECEnv): Environment.

        Returns:
            None.
        """

        self.env = env

    def get_nodes_static_feat_dim(self):
        """
        Nodes static features dimensions.

        Args:
            n/a.

        Returns:
            int: Nodes static features dimensions.
        """
        return sum(
            [
                self.feature_list.get("nodes_static", []).get(f).get("dim", 1)
                for f in self.feature_list.get("nodes_static")
            ]
        )

    def get_nodes_dynamic_feat_dim(self):
        """
        Nodes dynamic features dimensions.

        Args:
            n/a.

        Returns:
            int: Nodes dynamic features dimensions.
        """
        return len(self.feature_list.get("nodes_dynamic", []))

    def get_nodes_feat_dim(self):
        """
        Nodes features dimensions.

        Args:
            n/a.

        Returns:
            int: Nodes features dimensions.
        """
        return self.get_nodes_static_feat_dim() + self.get_nodes_dynamic_feat_dim()

    def get_agent_feat_dim(self):
        """
        Agent features dimensions.

        Args:
            n/a.

        Returns:
            int: Agent features dimensions.
        """
        return len(self.feature_list.get("agent", []))

    def get_other_agents_feat_dim(self):
        """
        Other agent features dimensions.

        Args:
            n/a.

        Returns:
            int: Other agent features dimensions.
        """
        return len(self.feature_list.get("other_agents", []))

    def get_all_agents_feat_dim(self):
        """
        Returns:
            int: all agents features dimentions.
        """
        return len(self.feature_list.get("all_agents", []))

    def get_global_feat_dim(self):
        """
        Global features dimensions.

        Args:
            n/a.

        Returns:
            int: Global features dimensions.
        """
        return len(self.feature_list.get("global", []))

    def compute_static_features(self):
        """
        Compute nodes static features.

        Args:
            n/a.

        Returns:
            torch.Tensor: Nodes static features.
        """
        features_static = self.feature_list.get("nodes_static")
        features_static_set = set(
            [features_static.get(f).get("feat") for f in features_static]
        )
        undefined_feat = features_static_set - set(self.possible_nodes_static_features)
        assert_msg = f"{undefined_feat} are not defined, choose from {str(self.possible_nodes_static_features)}"
        assert len(undefined_feat) == 0, assert_msg

        features = list()
        for f in features_static:
            f_feat = features_static.get(f).get("feat")
            dim = features_static.get(f).get("dim")
            if dim:
                feature = eval(f"self.get_feat_{f_feat}")(dim)
            else:
                feature = eval(f"self.get_feat_{f_feat}")()
            f_norm = features_static.get(f).get("norm")
            norm_feature = self._normalize_feature(feature, f_norm)
            features.append(norm_feature)
        return self._concat_features(features)

    def compute_dynamic_features(self):
        """
        Compute nodes dynamic features.

        Args:
            n/a.

        Returns:
            torch.Tensor: Nodes dynamic features.
        """
        features_dynamic = self.feature_list.get("nodes_dynamic")
        undefined_feat = set(features_dynamic) - set(
            self.possible_nodes_dynamic_features
        )
        assert_msg = f"{undefined_feat} are not defined, choose from {str(self.possible_nodes_dynamic_features)}"
        assert len(undefined_feat) == 0, assert_msg
        features = list()
        for f in features_dynamic:
            features.append(eval(f"self.get_feat_{f}")())
        return self._concat_features(features)

    def compute_agent_features(self):
        """
        Compute current agent features.

        Args:
            n/a.

        Returns:
            torch.Tensor: Current agent features.
        """
        features_self = self.feature_list.get("agent")
        undefined_feat = set(features_self) - set(self.possible_agent_features)
        assert_msg = f"{undefined_feat} are not defined, choose from {str(self.possible_agent_features)}"
        assert len(undefined_feat) == 0, assert_msg
        features = list()
        for f in features_self:
            features.append(eval(f"self.get_feat_agent_{f}")())
        return self._concat_features(features).squeeze(1)

    def compute_other_agents_features(self):
        """
        Compute other agent features.

        Args:
            n/a.

        Returns:
            torch.Tensor: Other agent features.
        """
        features_agents = self.feature_list.get("other_agents")
        undefined_feat = set(features_agents) - set(self.possible_other_agents_features)
        assert_msg = f"{undefined_feat} are not defined, choose from {str(self.possible_other_agents_features)}"
        assert len(undefined_feat) == 0, assert_msg
        features = list()
        for f in features_agents:
            features.append(eval(f"self.get_feat_other_agents_{f}")())
        return self._concat_features(features)

    def compute_all_agents_features(self):
        """
        Compute all agent features.

        Args:
            n/a.

        Returns:
            torch.Tensor: All agent features.
        """
        features_agents = self.feature_list.get("all_agents")
        undefined_feat = set(features_agents) - set(self.possible_all_agents_features)
        assert_msg = f"{undefined_feat} are not defined, choose from {str(self.possible_all_agents_features)}"
        assert len(undefined_feat) == 0, assert_msg
        features = list()
        for f in features_agents:
            features.append(eval(f"self.get_feat_all_agents_{f}")())
        return self._concat_features(features)

    def compute_global_features(self):
        """
        Compute global features.

        Args:
            n/a.

        Returns:
            torch.Tensor: Global features.
        """
        features_global = self.feature_list.get("global")
        undefined_feat = set(features_global) - set(self.possible_global_features)
        assert_msg = f"{undefined_feat} are not defined, choose from {str(self.possible_global_features)}"
        assert len(undefined_feat) == 0, assert_msg
        features = list()
        for f in features_global:
            features.append(eval(f"self.get_feat_global_{f}")())
        return self._concat_features(features).squeeze(dim=1)

    def get_observations(self, obs_list=None) -> TensorDict:
        """
        Get observations method.

        Args:
            obs_list: List of observations to compute. Defaults to None.

        Returns
            observations(TensorDict): Current environment observations and masks dictionary.
        """
        observations = TensorDict(
            {}, batch_size=self.env.batch_size, device=self.env.device
        )
        if obs_list is None:
            obs_list = self.feature_list.keys()

        if self.feature_list.get("nodes_static") and "nodes_static" in obs_list:
            static_feat = self.compute_static_features()
            observations["nodes_static_obs"] = static_feat

        if self.feature_list.get("nodes_dynamic") and "nodes_dynamic" in obs_list:
            dynamic_feat = self.compute_dynamic_features()
            observations["nodes_dynamic_obs"] = dynamic_feat

        if self.feature_list.get("agent") and "agent" in obs_list:
            agent_feat = self.compute_agent_features()
            observations["agent_obs"] = agent_feat

        if self.feature_list.get("other_agents") and "other_agents" in obs_list:
            agents_feat = self.compute_other_agents_features()
            mask_agents_feat = (
                self.env.td_state["agents"]["active_agents_mask"].unsqueeze(dim=-1)
                * agents_feat
            )
            observations["other_agents_obs"] = mask_agents_feat

        if self.feature_list.get("all_agents") and "all_agents" in obs_list:
            agents_feat = self.compute_all_agents_features()
            mask_agents_feat = (
                self.env.td_state["agents"]["active_agents_mask"].unsqueeze(dim=-1)
                * agents_feat
            )
            observations["all_agents_obs"] = mask_agents_feat

        if self.feature_list.get("global") and "global" in obs_list:
            global_feat = self.compute_global_features()
            observations["global_obs"] = global_feat

        return observations

    # auxiliary functions

    @staticmethod
    def _concat_features(features):
        """
        Concatenate features.

        Args:
            features(list): Features to concatenate.

        Returns:
            torch.Tensor: Concatenated tensor.
        """

        return torch.cat(
            [f.unsqueeze(dim=-1) if f.dim() == 2 else f for f in features], dim=-1
        )

    def _normalize_feature(self, x, norm):
        """
        Normalize features.

        Args:
            x(torch.Tensor): Tensor to be normalized.
            norm(str): Type of normalization. It can be 'min_max' or 'standardize'. If None, tensor is returned.

        Returns:
            torch.Tensor: Tensor normalized or default tensor if norm is invalid.
        """

        if norm == "min_max":
            return self._min_max_normalization(x)
        elif norm == "standardize":
            return self._standardize(x)
        elif norm == None:
            return x

    @staticmethod
    def _min_max_normalization(x):
        """
        Min. max. normalization.

        Args:
            x(torch.Tensor): Tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        max_x = torch.max(x, dim=1, keepdim=True)[0]
        min_x = torch.min(x, dim=1, keepdim=True)[0]
        return (x - min_x) / (max_x - min_x)

    @staticmethod
    def _min_max_normalization2d(x):
        """
        Min. max. normalization 2 dimensions.

        Args:
            x(torch.Tensor): Tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        max_x = torch.max(x)
        min_x = torch.min(x)
        return (x - min_x) / (max_x - min_x)

    @staticmethod
    def _standardize(x):
        """
        Tensor standardization.

        Args:
            x(torch.Tensor): Tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        means = x.mean(dim=1, keepdim=True)
        stds = x.std(dim=1, keepdim=True)
        return (x - means) / stds
