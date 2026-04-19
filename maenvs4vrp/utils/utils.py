import torch
from tensordict import TensorDict
from torch import Tensor
from typing import Any, Optional, Tuple, List, Dict


def data_equivalence(data_1, data_2, exact: bool = False) -> bool:
    # adapted from https://gymnasium.farama.org/main/_modules/gymnasium/utils/env_checker/
    """Assert equality between data 1 and 2, i.e observations, actions, info.

    Args:
        data_1: Data structure 1
        data_2: Data structure 2
        exact: Whether to compare array exactly or not if false compares with absolute and realive torrelance of 1e-5 (for more information check [np.allclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html)).

    Returns:
        If observation 1 and 2 are equivalent
    """
    if type(data_1) is not type(data_2):
        return False
    if isinstance(data_1, dict) or isinstance(data_1, TensorDict):
        return data_1.keys() == data_2.keys() and all(
            data_equivalence(data_1[k], data_2[k], exact) for k in data_1.keys()
        )
    elif isinstance(data_1, (tuple, list)):
        return len(data_1) == len(data_2) and all(
            data_equivalence(o_1, o_2, exact) for o_1, o_2 in zip(data_1, data_2)
        )
    elif isinstance(data_1, Tensor):
        if data_1.shape == data_2.shape and data_1.dtype == data_2.dtype:
            if data_1.dtype == object:
                return all(
                    data_equivalence(a, b, exact) for a, b in zip(data_1, data_2)
                )
            else:
                if exact:
                    return torch.all(data_1 == data_2)
                else:
                    return torch.allclose(data_1, data_2, rtol=1e-5, atol=1e-5)
        else:
            return False
    else:
        return data_1 == data_2


def get_solution(
    env: Any,
    batch_idx: Optional[int] = None,
    include_depot: bool = True,
    drop_empty_tours: bool = True,
) -> Dict[str, Any] | List[Dict[str, Any]]:
    """
    Build a normalized solution dictionary from env.td_state.

    Always returns a dict (or list of dicts if batch_idx=None), never None.
    Adds:
      - agent_depot: {agent_id: depot_node_idx or None}
      - depot_agents: {depot_node_idx: [agent_ids]}
    """
    assert hasattr(
        env, "td_state"
    ), "Environment has no td_state. Did you call env.reset()?"
    td = env.td_state

    # Determine batch size (B)
    if "coords" in td.keys():
        B = int(td["coords"].shape[0])
    else:
        bs = getattr(env, "batch_size", torch.Size([1]))
        B = int(
            bs.numel()
            if isinstance(bs, torch.Size) and len(bs) > 1
            else (bs[0] if isinstance(bs, torch.Size) and len(bs) == 1 else 1)
        )

    # Extract event stream if present
    actions: Optional[torch.Tensor] = None
    agents: Optional[torch.Tensor] = None
    T = 0
    if "solution" in td.keys():
        sol_td = td["solution"]
        if "actions" in sol_td.keys():
            actions = sol_td["actions"]
            if actions.dim() == 3:
                actions = actions.squeeze(-1)
        if "agents" in sol_td.keys():
            agents = sol_td["agents"]
            if agents.dim() == 3:
                agents = agents.squeeze(-1)
        if isinstance(actions, torch.Tensor):
            T = int(actions.shape[-1])

    # Depot information (mask and/or index)
    is_depot_full: Optional[torch.Tensor] = (
        td["is_depot"] if "is_depot" in td.keys() else None
    )  # [B, N] bool
    depot_idx_tensor: Optional[torch.Tensor] = None
    if "depot_idx" in td.keys():
        dep_idx = td["depot_idx"]
        if dep_idx.dim() == 1:
            depot_idx_tensor = dep_idx
        elif dep_idx.dim() == 2 and dep_idx.shape[1] == 1:
            depot_idx_tensor = dep_idx.squeeze(1)
        else:
            depot_idx_tensor = None  # multi-depot shape

    def build_for_batch(b: int) -> Dict[str, Any]:
        # Depots for this batch item
        depots_b: List[int] = []
        depot_single: Optional[int] = None
        is_dep_b: Optional[torch.Tensor] = None

        if isinstance(is_depot_full, torch.Tensor):
            is_dep_b = is_depot_full[b].to(dtype=torch.bool)
            depots_b = torch.where(is_dep_b)[0].detach().cpu().tolist()
            if len(depots_b) == 1:
                depot_single = depots_b[0]

        if depot_single is None and depot_idx_tensor is not None:
            di = depot_idx_tensor[b]
            di = int(di.item()) if isinstance(di, torch.Tensor) else int(di)
            depot_single = di
            # synthesize a minimal mask if we know coords
            if is_dep_b is None and "coords" in td.keys():
                N = int(td["coords"].shape[1])
                device = (
                    actions.device if isinstance(actions, torch.Tensor) else td.device
                )
                is_dep_b = torch.zeros(N, dtype=torch.bool, device=device)
                if 0 <= depot_single < N:
                    is_dep_b[depot_single] = True

        # Helper: check depot node
        if is_dep_b is None:

            def is_depot_node(node_idx: int) -> bool:
                return depot_single is not None and node_idx == depot_single

        else:
            dep_mask_np = is_dep_b.detach().cpu()

            def is_depot_node(node_idx: int) -> bool:
                if 0 <= node_idx < dep_mask_np.numel():
                    return bool(dep_mask_np[node_idx].item())
                return depot_single is not None and node_idx == depot_single

        # Agent -> depot assignment
        agent_depot_map: Dict[int, Optional[int]] = {}
        num_agents_env = int(getattr(env, "num_agents", 1))
        if "agents" in td.keys() and "depot_idx" in td["agents"].keys():
            dep_idx_per_agent = td["agents"]["depot_idx"][b]
            num_agents_here = (
                int(dep_idx_per_agent.shape[0])
                if dep_idx_per_agent.ndim >= 1
                else num_agents_env
            )
            for a in range(num_agents_here):
                agent_depot_map[a] = int(dep_idx_per_agent[a].item())
        else:
            # Fallbacks
            for a in range(num_agents_env):
                agent_depot_map[a] = (
                    int(depot_single) if depot_single is not None else None
                )

        # Inverse mapping: depot -> [agents]
        depot_agents: Dict[int, List[int]] = {}
        for a, d in agent_depot_map.items():
            if d is not None:
                depot_agents.setdefault(d, []).append(a)

        # If no actions yet, return a minimal, valid structure
        if actions is None or agents is None or T == 0:
            num_agents_here = int(getattr(env, "num_agents", len(agent_depot_map) or 1))
            if len(depots_b) == 0 and depot_single is not None:
                depots_b = [int(depot_single)]
            tours: Dict[int, List[List[int]]] = {a: [] for a in range(num_agents_here)}
            edges: Dict[int, List[Tuple[int, int]]] = {
                a: [] for a in range(num_agents_here)
            }
            return {
                "depot": depot_single,
                "depots": depots_b,
                "tours": tours,
                "edges": edges,
                "agent_depot": agent_depot_map,
                "depot_agents": depot_agents,
            }

        # Build tours by replaying the event stream
        tours: Dict[int, List[List[int]]] = {a: [] for a in range(num_agents_env)}
        current: Dict[int, List[int]] = {}
        start_depot: Dict[int, Optional[int]] = {a: None for a in range(num_agents_env)}

        # Seed with agent's assigned depot if include_depot
        for a in range(num_agents_env):
            if include_depot:
                a_dep = agent_depot_map.get(a, None)
                if a_dep is not None:
                    current[a] = [int(a_dep)]
                    start_depot[a] = int(a_dep)
                elif depot_single is not None:
                    current[a] = [int(depot_single)]
                    start_depot[a] = int(depot_single)
                else:
                    current[a] = []
            else:
                current[a] = []

        # Replay actions
        for t in range(T):
            agent_id = int(agents[b, t].item())
            node = int(actions[b, t].item())
            if is_depot_node(node):
                # Close current tour if it has visits beyond initial depot (when include_depot)
                if len(current[agent_id]) > (1 if include_depot else 0):
                    if include_depot and (
                        len(current[agent_id]) == 0
                        or not is_depot_node(current[agent_id][-1])
                    ):
                        current[agent_id].append(node)
                    tours[agent_id].append(current[agent_id])
                # Start next tour
                if include_depot:
                    current[agent_id] = [node]
                    start_depot[agent_id] = node
                else:
                    current[agent_id] = []
                    start_depot[agent_id] = node
            else:
                current[agent_id].append(node)

        # Helper to infer start depot if needed
        def infer_start_depot_for_agent(a: int, tour: List[int]) -> Optional[int]:
            if agent_depot_map.get(a, None) is not None:
                return int(agent_depot_map[a])
            if start_depot[a] is not None:
                return int(start_depot[a])
            if depot_single is not None:
                return int(depot_single)
            if len(depots_b) > 0 and "coords" in td.keys() and len(tour) > 0:
                coords_b = td["coords"][b]
                first = tour[0]
                if 0 <= first < coords_b.shape[0]:
                    p = coords_b[first].unsqueeze(0)
                    dep_pts = coords_b[torch.tensor(depots_b, device=coords_b.device)]
                    dists = torch.cdist(p, dep_pts, p=2).squeeze(0)
                    idx = int(torch.argmin(dists).item())
                    return int(depots_b[idx])
            if len(depots_b) > 0:
                return int(depots_b[0])
            return None

        # Flush unfinished tours and ensure depot at start/end
        for a in range(num_agents_env):
            tour = current[a]
            if len(tour) > (1 if include_depot else 0):
                if include_depot:
                    a_dep = agent_depot_map.get(a, None)
                    if a_dep is not None and (len(tour) == 0 or tour[0] != a_dep):
                        if len(tour) == 0 or not is_depot_node(tour[0]):
                            tour = [int(a_dep)] + tour
                    elif not is_depot_node(tour[0]):
                        sd = infer_start_depot_for_agent(a, tour)
                        if sd is not None:
                            tour = [sd] + tour

                    if a_dep is not None:
                        if not is_depot_node(tour[-1]) or tour[-1] != a_dep:
                            tour = tour + [int(a_dep)]
                    elif not is_depot_node(tour[-1]):
                        if start_depot[a] is not None:
                            tour = tour + [int(start_depot[a])]
                        elif depot_single is not None:
                            tour = tour + [int(depot_single)]
                        elif len(depots_b) > 0:
                            if len(tour) > 0 and is_depot_node(tour[0]):
                                tour = tour + [int(tour[0])]
                            else:
                                sd = infer_start_depot_for_agent(a, tour)
                                if sd is not None:
                                    tour = tour + [sd]
                tours[a].append(tour)

        # Build edges
        edges: Dict[int, List[Tuple[int, int]]] = {a: [] for a in range(num_agents_env)}
        for a, agent_tours in tours.items():
            for tour in agent_tours:
                for i in range(len(tour) - 1):
                    edges[a].append((tour[i], tour[i + 1]))

        return {
            "depot": depot_single,
            "depots": depots_b,
            "tours": tours,
            "edges": edges,
            "agent_depot": agent_depot_map,
            "depot_agents": depot_agents,
        }

    if batch_idx is None:
        return [build_for_batch(b) for b in range(B)]
    else:
        assert 0 <= batch_idx < B, f"batch_idx {batch_idx} out of range [0, {B})"
        return build_for_batch(batch_idx)
