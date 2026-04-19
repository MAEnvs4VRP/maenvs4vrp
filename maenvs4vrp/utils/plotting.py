# Python
from __future__ import annotations
from typing import Any, Mapping, Optional, Sequence, Tuple, List, Dict
import math
import torch
from maenvs4vrp.core.env import AECEnv
import matplotlib.pyplot as plt
from maenvs4vrp.utils.utils import get_solution


def plot_instance_coords(
    instance: Mapping[str, Any],
    batch_idx: int = 0,
    annotate: bool = True,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (5, 5),
    point_size: int | float = 50,
    show_depot: bool = True,
    show_legend: bool = True,
    ax: "Optional[object]" = None,  # Matplotlib Axes, kept as object to avoid hard import at module scope
) -> None:
    """
    Plot node coordinates from an instance, optionally highlighting the depot(s).

    The function infers depot information in this order:
      1) instance['data']['is_depot'] -> boolean tensor mask (batched or not)
      2) instance['data']['depot_idx'] or instance['depot_idx'] -> int index
      3) fallback: depot index 0

    Args:
        instance(Mapping[str, Any]): Mapping with a 'data' key containing 'coords' tensor of shape [B, N, 2] or [N, 2].
        batch_idx(int): Which batch to plot if coords are batched. Defaults to 0.
        annotate(bool): Whether to label points with their indices. Defaults to True.
        title(str, optional): Plot title. Defaults to instance.get("name", "Instance") if None. Defaults to None.
        figsize(tuple[int, int]): Figure size for matplotlib (used only when ax is None). Defaults to (5, 5).
        point_size(int or float): Marker size for points. Defaults to 50.
        show_depot(bool): If True, highlight depot(s). If False, plot all points uniformly. Defaults to True.
        show_legend(bool): If True, show legend. Defaults to True.
        ax(object, optional): Optional Matplotlib Axes to draw on. If None, a figure is created and shown. Defaults to None.

    Returns:
        None.
    """
    coords = instance["data"]["coords"]

    # Normalize coords to [N, 2]
    if isinstance(coords, torch.Tensor) and coords.dim() == 3:
        xy = coords[batch_idx]
    elif isinstance(coords, torch.Tensor) and coords.dim() == 2:
        xy = coords
    else:
        raise ValueError("coords must be a torch.Tensor of shape [B, N, 2] or [N, 2].")

    xy = xy.detach().cpu()
    N = xy.shape[0]

    # Build depot mask
    depot_mask = torch.zeros(N, dtype=torch.bool)

    if show_depot:
        is_depot = None
        # Try is_depot mask first
        if (
            isinstance(instance.get("data", {}), Mapping)
            and "is_depot" in instance["data"]
        ):
            is_depot_raw = instance["data"]["is_depot"]
            if isinstance(is_depot_raw, torch.Tensor):
                if is_depot_raw.dim() == 2:
                    is_depot = is_depot_raw[batch_idx].to(dtype=torch.bool)
                elif is_depot_raw.dim() == 1:
                    is_depot = is_depot_raw.to(dtype=torch.bool)
        if is_depot is not None:
            depot_mask = is_depot.detach().cpu()
        else:
            # Try single depot index
            depot_idx = None
            data = instance.get("data", {})
            if isinstance(data, Mapping) and "depot_idx" in data:
                depot_idx = (
                    int(data["depot_idx"])
                    if not isinstance(data["depot_idx"], torch.Tensor)
                    else int(data["depot_idx"].detach().cpu().item())
                )
            elif "depot_idx" in instance:
                depot_idx_val = instance["depot_idx"]
                depot_idx = (
                    int(depot_idx_val)
                    if not isinstance(depot_idx_val, torch.Tensor)
                    else int(depot_idx_val.detach().cpu().item())
                )

            if depot_idx is None:
                # Fallback: assume 0 is depot if available
                depot_idx = 0 if N > 0 else None

            if depot_idx is not None and 0 <= depot_idx < N:
                depot_mask[depot_idx] = True

    services_mask = ~depot_mask if show_depot else torch.ones(N, dtype=torch.bool)

    # Prepare axes
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=figsize)

    # Services or all points (when show_depot=False)
    if services_mask.any():
        ax.scatter(
            xy[services_mask, 0],
            xy[services_mask, 1],
            s=point_size,
            c="#1f77b4",
            label="Service" if show_depot else "Node",
        )
    # Depot(s)
    if show_depot and depot_mask.any():
        ax.scatter(
            xy[depot_mask, 0],
            xy[depot_mask, 1],
            s=point_size * 1.2,
            c="#d62728",
            marker="s",
            label="Depot",
        )

    if annotate:
        for i, (x, y) in enumerate(xy.tolist()):
            ax.annotate(
                str(i), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8
            )

    ax.set_aspect("equal", adjustable="box")
    if N > 0:
        ax.set_xlim(float(xy[:, 0].min()) - 0.02, float(xy[:, 0].max()) + 0.02)
        ax.set_ylim(float(xy[:, 1].min()) - 0.02, float(xy[:, 1].max()) + 0.02)
    ax.grid(True, alpha=0.2)
    ax.set_title(title or instance.get("name", "Instance"))

    # Safe legend handling
    if show_legend:
        # Place legend below the plot and reserve bottom space
        handles, labels = ax.get_legend_handles_labels()
        # Filter out empty/private labels just in case
        filtered = [
            (h, l) for h, l in zip(handles, labels) if l and not str(l).startswith("_")
        ]
        if filtered:
            handles, labels = zip(*filtered)
            ncol = max(1, min(3, len(labels)))
            leg = ax.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                ncol=ncol,
                borderaxespad=0.0,
                frameon=True,
            )
            fig = ax.figure
            try:
                # Leave space at the bottom for the outside legend
                fig.subplots_adjust(bottom=0.22)
            except Exception:
                pass
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # Only show if we created the figure
    if created_fig is not None:
        plt.show()


def plot_random_batch_instances(
    instance: Mapping[str, Any],
    n: int,
    seed: int = 0,
    cols: Optional[int] = None,
    annotate: bool = True,
    show_depot: bool = True,
    point_size: int | float = 50,
    figsize_per_plot: Tuple[float, float] = (4.0, 4.0),
    titles: Optional[Sequence[str]] = None,
    show: bool = True,
    return_objects: bool = False,
) -> Optional[Tuple["object", "List[object]", List[int]]]:
    """
    Plot a seeded random selection of n items from a batched instance into a grid of subplots.

    Expected input:
        instance: Mapping with key "data" -> "coords" as a torch.Tensor of shape [B, N, 2], where B is batch size, N is number of nodes, and each row is (x, y).

    Parameters
        n(int): Number of batch items (from B) to plot. Must satisfy 1 <= n <= B.
        seed(int): Random seed used for the selection without replacement. Defaults to 0.
        cols(int, optional): Number of columns in the subplot grid. If None, a near-square grid is chosen. Defaults to None.
        annotate(bool): If True, annotate nodes with their indices. Defaults to True.
        show_depot(bool): If True, highlight depot node(s) when present. Defaults to True.
        point_size(int or float): Marker size for node scatter plots. Defaults to 50.
        figsize_per_plot(Tuple[float, float]): (width, height) for each subplot; total figure size scales by the grid. Defaults to (4.0, 4.0).
        titles(Sequence[str], optional): Optional sequence of length >= n to use as subplot titles; otherwise defaults to "Batch {idx}" for each selected item. Defaults to None.
        show(bool): If True, plt.show() is called inside the function. Defaults to True.
        return_objects(bool): If True, returns (fig, axes_list, selected_indices); if False (default), returns None. Defaults to False.

    Returns
        (fig, axes_list, selected_indices) when return_objects=True:
            * fig: matplotlib.figure.Figure
            * axes_list: list of Axes for the n plotted items (flattened, length n)
            * selected_indices: sorted list of the selected batch indices
        Otherwise returns None.

    Errors:
        ValueError if coords is missing, not a 3D torch.Tensor, or has incompatible shape.
        ValueError if n < 1 or n > B.

    Notes
        The function creates one shared legend for the entire figure when labels are present.
        Layout is tightened; additional padding is added when a shared legend is drawn.
    """
    coords = instance["data"]["coords"]
    if not (isinstance(coords, torch.Tensor) and coords.dim() == 3):
        raise ValueError("Expected batched coords with shape [B, N, 2].")

    B = coords.shape[0]
    if n < 1:
        raise ValueError("n must be >= 1.")
    if n > B:
        raise ValueError(f"n ({n}) cannot exceed batch size ({B}).")

    # Seeded random selection without replacement
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(B, generator=g).tolist()
    selected = perm[:n]
    selected.sort()

    # Grid layout
    if cols is None:
        cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    fig_w = figsize_per_plot[0] * cols
    fig_h = figsize_per_plot[1] * rows
    fig, axs = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
    axs_flat: List[object] = [ax for row in axs for ax in row]

    # Plot each selected batch element
    for i, batch_idx in enumerate(selected):
        ax = axs_flat[i]
        ttl = (
            titles[i]
            if (titles is not None and i < len(titles))
            else f"Batch {batch_idx}"
        )
        plot_instance_coords(
            instance=instance,
            batch_idx=batch_idx,
            annotate=annotate,
            title=ttl,
            figsize=figsize_per_plot,
            point_size=point_size,
            show_depot=show_depot,
            show_legend=False,
            ax=ax,
        )

    # Hide any unused axes if grid has extra cells
    for j in range(n, rows * cols):
        axs_flat[j].axis("off")

    # Create a single, shared legend for the whole figure
    legend_added = False
    if n > 0:
        handles, labels = axs_flat[0].get_legend_handles_labels()
        filtered = [
            (h, l) for h, l in zip(handles, labels) if l and not str(l).startswith("_")
        ]
        if filtered:
            handles, labels = zip(*filtered)
            ncol = max(1, min(3, len(labels)))
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.005),
                ncol=ncol,
                borderaxespad=0.0,
                frameon=True,
            )
            legend_added = True
            try:
                fig.subplots_adjust(bottom=0.06)
            except Exception:
                pass

    if legend_added:
        try:
            fig.tight_layout(rect=(0.0, 0.02, 1.0, 1.0))
        except Exception:
            pass
    else:
        fig.tight_layout()

    if show:
        plt.show()

    if return_objects:
        return fig, axs_flat[:n], selected
    return None


def plot_env_instance_coords(
    env: AECEnv,
    batch_idx: int = 0,
    annotate: bool = True,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (5, 5),
    point_size: int | float = 50,
    show_depot: bool = True,
    show_legend: bool = True,
    ax: "Optional[object]" = None,
    # Matplotlib Axes, kept as object to avoid hard import at module scope
) -> None:
    instance = env._get_current_instance_data()
    plot_instance_coords(
        instance=instance,
        batch_idx=batch_idx,
        annotate=annotate,
        title=title,
        figsize=figsize,
        point_size=point_size,
        show_depot=show_depot,
        show_legend=show_legend,
        ax=ax,
    )


def plot_env_random_batch_instances(
    env: AECEnv,
    n: int,
    seed: int = 0,
    cols: Optional[int] = None,
    annotate: bool = True,
    show_depot: bool = True,
    point_size: int | float = 50,
    figsize_per_plot: Tuple[float, float] = (4.0, 4.0),
    titles: Optional[Sequence[str]] = None,
    show: bool = True,
    return_objects: bool = False,
) -> Optional[Tuple["object", "List[object]", List[int]]]:
    """
    Plot a seeded random selection of n items from the current environment's batched instance.

    This is a convenience wrapper around plot_random_batch_instances that pulls the
    active instance data from env._get_current_instance_data().

    Args:
        env(AECEnv): Environment providing the current batched instance.
        n(int): Number of batch items to plot (1 <= n <= batch size).
        seed(int): Seed for reproducible selection without replacement. Defaults to 0.
        cols(int, optional): Columns in the subplot grid. If None, a near-square layout is chosen. Defaults to None.
        annotate(bool): If True, annotate nodes with their indices. Defaults to True.
        show_depot(bool): If True, highlight depot node(s) when present. Defaults to True.
        point_size(int or float): Marker size for nodes. Defaults to 50.
        figsize_per_plot(Tuple[float, float]): (width, height) for each subplot. Defaults to (4.0, 4.0).
        titles(Sequence[str]): Optional titles per subplot; falls back to "Batch {idx}". Defaults to None.
        show(bool): If True, calls plt.show() inside the function. Default to True.
        return_objects(bool): If True, returns (fig, axes_list, selected_indices); if False (default), returns None. Defaults to False.

    Returns:
        (fig, axes_list, selected_indices) when return_objects=True; otherwise None.
    """
    instance = env._get_current_instance_data()
    return plot_random_batch_instances(
        instance=instance,
        n=n,
        seed=seed,
        cols=cols,
        annotate=annotate,
        show_depot=show_depot,
        point_size=point_size,
        figsize_per_plot=figsize_per_plot,
        titles=titles,
        show=show,
        return_objects=return_objects,
    )


def plot_solution_overlay(
    ax: plt.Axes,
    coords: torch.Tensor,  # shape [N, 2], CPU or CUDA
    solution: Dict[str, Any],  # output of get_solution(..., batch_idx=...)
    colors: Optional[List[str]] = None,
    linewidth: float = 1.5,
    alpha: float = 0.9,
    show_depot: bool = True,
    depot_kwargs: Optional[Dict[str, Any]] = None,
    arrows: bool = False,
    arrowstyle: str = "-|>",
    mutation_scale: float = 12.0,
    arrow_every: int = 1,
) -> None:
    """
    Draw solution routes on top of an existing scatter plot of nodes.

    Args:
        ax(plt.Axes): Axes where solution will be drawn.
        coords(torch.Tensor): Coords with format [N, 2]. N nodes have 2 coordinates.
        solution(Dict[str, Any]): Output of get_solution.
        colors(List[str], optional): Color list for agents routes.
        linewidth(float): Width of drawn lines. Defaults to 1.5.
        alpha(float): Lines opacity. 0 is invisible and 1 is opaque. Defaults to 0.9.
        show_depot(bool): If True, it draws the depot. Defaults to True.
        depot_kwargs(Dict[str, Any], optional): Additional args for depot scatter. Defaults to None.
        arrows(bool): If True, routes are arrows. If False, routes are lines. Defaults to False.
        arrowstyle(str): Arrows style when arrows=True. Defaults to '-|>'.
        mutation_scale(float): Arrows scale when arrows=True. Defaults to 12.0.
        arrow_every(int): How often arrows are drawn. If 1, draws arrows in every line. If 2, draws arrows in every other line. etc. Defaults to 1.

    Returns:
        None.
    """
    xy = coords.detach().cpu().numpy()
    depot = solution["depot"]

    # Highlight depot if requested
    if show_depot and depot is not None:
        dk = {"s": 100, "c": "#d62728", "marker": "s", "label": "Depot"}
        dk.update(depot_kwargs or {})
        ax.scatter(xy[depot, 0], xy[depot, 1], **dk)

    # Prepare colors
    agent_ids = sorted(solution["edges"].keys())
    if colors is None or len(colors) == 0:
        colors = [
            "#1f77b4",
            "#2ca02c",
            "#ff7f0e",
            "#9467bd",
            "#8c564b",
            "#17becf",
            "#e377c2",
            "#7f7f7f",
        ]

    # Agent -> depot mapping (optional, for legend annotation)
    agent_depot = solution.get("agent_depot", {})

    # Draw edges per agent
    for i, a in enumerate(agent_ids):
        col = colors[i % len(colors)]
        segs = solution["edges"][a]
        if not segs:
            continue

        legend_label = f"Agent {a}"
        if (
            isinstance(agent_depot, dict)
            and a in agent_depot
            and agent_depot[a] is not None
        ):
            legend_label = f"{legend_label} (D {agent_depot[a]})"

        if not arrows:
            xs = []
            ys = []
            for u, v in segs:
                xs.extend([xy[u, 0], xy[v, 0], None])
                ys.extend([xy[u, 1], xy[v, 1], None])
            ax.plot(
                xs, ys, color=col, linewidth=linewidth, alpha=alpha, label=legend_label
            )
        else:
            drew_any = False
            for j, (u, v) in enumerate(segs):
                if arrow_every > 1 and (j % arrow_every) != 0:
                    continue
                ax.annotate(
                    "",
                    xy=(xy[v, 0], xy[v, 1]),
                    xytext=(xy[u, 0], xy[u, 1]),
                    arrowprops=dict(
                        arrowstyle=arrowstyle,
                        color=col,
                        linewidth=linewidth,
                        shrinkA=0,
                        shrinkB=0,
                        mutation_scale=mutation_scale,
                        alpha=alpha,
                    ),
                    zorder=3,
                )
                drew_any = True
            if drew_any:
                ax.plot(
                    [],
                    [],
                    color=col,
                    linewidth=linewidth,
                    alpha=alpha,
                    label=legend_label,
                )


def plot_solution(
    env,
    batch_idx: int,
    annotate: bool = True,
    include_depot_edges: bool = True,
    per_depot_subplots: bool = False,
    cols: int | None = None,
    figsize_per_subplot: tuple[float, float] = (6.0, 6.0),
):
    """
    Plot the current environment's solution for a given batch item.

    The function renders either:
        A single axes view of all depots and routes (default), or
        One subplot per depot (when per_depot_subplots=True and multiple depots exist).

    Parameters
        env: Environment containing td_state and solution information.
        batch_idx(int): Index of the batch item to visualize.
        annotate(bool): If True, annotates nodes with their indices. Defaults to True.
        include_depot_edges(bool): If False, edges incident to depot nodes are omitted from overlay. Defaults to True.
        per_depot_subplots(bool): If True and multiple depots exist, creates a grid with one subplot per depot. Defaults to False.
        cols(int or None): Number of columns for the per-depot subplot grid (if None, a near-square layout is chosen). Defaults to None.
        figsize_per_subplot(tuple[float, float]): (width, height) size used for each subplot when per_depot_subplots=True. Defaults to (6.0, 6.0).

    Behavior
        If the solution is missing, a minimal empty solution is synthesized defensively.
        Axis limits, equal aspect, grid, and titles are set for readability.
        Legends are placed below axes (for per-depot subplots) or as a shared figure legend (single-axes case).

    Returns
        None. The function produces the plot and calls plt.show().
    """
    coords: torch.Tensor = env.td_state[batch_idx]["coords"]
    solution = get_solution(
        env, batch_idx=batch_idx, include_depot=True, drop_empty_tours=True
    )

    # Defensive: synthesize a minimal solution if any external caller returns/feeds None
    if solution is None:
        td_b = env.td_state[batch_idx]
        depots_b = []
        depot_single = None
        if "is_depot" in td_b.keys():
            is_dep = td_b["is_depot"].to(dtype=torch.bool)
            depots_b = torch.where(is_dep)[0].detach().cpu().tolist()
        elif "depot_idx" in td_b.keys():
            depots_b = td_b["depot_idx"].detach().cpu().view(-1).tolist()
        if len(depots_b) == 1:
            depot_single = depots_b[0]
        num_agents = getattr(env, "num_agents", 1)
        solution = {
            "depot": depot_single,
            "depots": depots_b,
            "tours": {a: [] for a in range(num_agents)},
            "edges": {a: [] for a in range(num_agents)},
            "agent_depot": {
                a: (depot_single if depot_single is not None else None)
                for a in range(num_agents)
            },
        }

    # Determine a human-friendly instance name for titles
    instance_name = getattr(env, "instance_name", None)
    if instance_name is None:
        try:
            if hasattr(env, "_get_current_instance_data"):
                _inst = env._get_current_instance_data()
                if isinstance(_inst, dict) and "name" in _inst:
                    instance_name = _inst["name"]
        except Exception:
            instance_name = None
    if instance_name is None:
        instance_name = getattr(env, "env_name", "Instance")

    xy = coords.detach().cpu()
    N = xy.shape[0]

    # Handle both single and multi-depot outputs from get_solution
    depots_list = solution.get("depots", None)
    single_depot = solution.get("depot", None)
    agent_depot_map: Dict[int, Optional[int]] = solution.get("agent_depot", {})

    # If requested, create one subplot per depot (only if multi-depot and we know agent assignments)
    if per_depot_subplots:
        depots = []
        if isinstance(depots_list, list) and len(depots_list) > 0:
            depots = depots_list
        elif isinstance(single_depot, int):
            depots = [single_depot]

        if (
            len(depots) > 1
            and isinstance(agent_depot_map, dict)
            and len(agent_depot_map) > 0
        ):
            # Create grid
            D = len(depots)
            if cols is None or cols <= 0:
                # near-square layout
                import math  # local import to avoid global dependency

                cols = int(math.ceil(math.sqrt(D)))
            rows = (D + cols - 1) // cols
            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(figsize_per_subplot[0] * cols, figsize_per_subplot[1] * rows),
                squeeze=False,
            )

            for idx, d in enumerate(depots):
                r, c = divmod(idx, cols)
                ax = axes[r][c]

                # Build depot mask for this subplot
                depot_mask = torch.zeros(N, dtype=torch.bool)
                if isinstance(d, int) and 0 <= d < N:
                    depot_mask[d] = True

                services_mask = (
                    ~depot_mask if depot_mask.any() else torch.ones(N, dtype=torch.bool)
                )

                # Plot services
                if services_mask.any():
                    ax.scatter(
                        xy[services_mask, 0],
                        xy[services_mask, 1],
                        s=24,
                        c="#1f77b4",
                        label="Service",
                    )

                # Plot the current depot only
                if depot_mask.any():
                    ax.scatter(
                        xy[depot_mask, 0],
                        xy[depot_mask, 1],
                        s=110,
                        c="#d62728",
                        marker="s",
                        label=f"Depot {d}",
                    )

                # Prepare filtered solution for this depot: only agents assigned to d
                agents_for_d = [a for a, dep in agent_depot_map.items() if dep == d]
                filtered_edges: Dict[int, List[Tuple[int, int]]] = {
                    a: solution["edges"].get(a, []) for a in agents_for_d
                }
                filtered_tours: Dict[int, List[List[int]]] = {
                    a: solution["tours"].get(a, []) for a in agents_for_d
                }
                sol_d = {
                    "depot": d,
                    "depots": [d],
                    "tours": filtered_tours,
                    "edges": filtered_edges,
                    "agent_depot": {a: d for a in agents_for_d},
                }

                # Optionally filter out edges that touch depot nodes (for this depot)
                if not include_depot_edges:
                    depots_set = {d}
                    fe: Dict[int, List[Tuple[int, int]]] = {}
                    for a, edges in sol_d["edges"].items():
                        fe[a] = [
                            (u, v)
                            for (u, v) in edges
                            if (u not in depots_set and v not in depots_set)
                        ]
                    sol_d = {**sol_d, "edges": fe}

                # Overlay solution routes
                plot_solution_overlay(
                    ax=ax,
                    coords=coords,
                    solution=sol_d,
                    show_depot=False,  # we already plotted the depot marker above
                    arrows=True,
                    arrowstyle="-|>",
                    mutation_scale=12.0,
                    arrow_every=1,
                )

                # Optional annotations
                if annotate:
                    for i, (x, y) in enumerate(xy.tolist()):
                        ax.annotate(
                            str(i),
                            (x, y),
                            textcoords="offset points",
                            xytext=(4, 4),
                            fontsize=8,
                        )

                ax.set_aspect("equal", adjustable="box")
                if N > 0:
                    ax.set_xlim(
                        float(xy[:, 0].min()) - 0.02, float(xy[:, 0].max()) + 0.02
                    )
                    ax.set_ylim(
                        float(xy[:, 1].min()) - 0.02, float(xy[:, 1].max()) + 0.02
                    )
                ax.grid(True, alpha=0.2)

                # Subplot title includes instance name, depot and agent list
                ax.set_title(
                    f"{instance_name} — Depot {d}: Agents {agents_for_d if len(agents_for_d) > 0 else '[]'}"
                )

                # Legend below each subplot (close to the axes)
                try:
                    handles, labels = ax.get_legend_handles_labels()
                    filtered = [
                        (h, l)
                        for h, l in zip(handles, labels)
                        if l and not str(l).startswith("_")
                    ]
                    if filtered:
                        handles, labels = zip(*filtered)
                        ncol = max(1, min(4, len(labels)))
                        ax.legend(
                            handles,
                            labels,
                            loc="upper center",
                            bbox_to_anchor=(0.5, -0.06),  # push legend further down
                            ncol=ncol,
                            borderaxespad=0.0,
                            frameon=True,
                        )
                except Exception:
                    pass

                # Hide any unused axes (AFTER the loop)
            for j in range(D, rows * cols):
                r, c = divmod(j, cols)
                axes[r][c].axis("off")

                # Increase vertical spacing so per-axes legends have room below each subplot (AFTER the loop)
            try:
                fig.subplots_adjust(hspace=10.0)  # more space between rows
            except Exception:
                pass

            plt.tight_layout()
            plt.show()
            return  # done with per-depot subplots

    # ===== Default single-axes behavior (original) =====
    depot_mask = torch.zeros(N, dtype=torch.bool)
    depots_set = set()
    if isinstance(depots_list, list) and len(depots_list) > 0:
        for d in depots_list:
            if isinstance(d, int) and 0 <= d < N:
                depot_mask[d] = True
                depots_set.add(d)
    elif isinstance(single_depot, int) and 0 <= single_depot < N:
        depot_mask[single_depot] = True
        depots_set.add(single_depot)

    services_mask = ~depot_mask if depot_mask.any() else torch.ones(N, dtype=torch.bool)

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot services
    if services_mask.any():
        ax.scatter(
            xy[services_mask, 0],
            xy[services_mask, 1],
            s=24,
            c="#1f77b4",
            label="Service",
        )

    # Plot depot(s)
    if depot_mask.any():
        ax.scatter(
            xy[depot_mask, 0],
            xy[depot_mask, 1],
            s=110,
            c="#d62728",
            marker="s",
            label="Depot",
        )

    # Optionally filter out edges that touch depot nodes
    solution_for_overlay = solution
    if not include_depot_edges and len(depots_set) > 0:
        filtered_edges: Dict[int, List[Tuple[int, int]]] = {}
        for a, edges in solution["edges"].items():
            filtered_edges[a] = [
                (u, v)
                for (u, v) in edges
                if (u not in depots_set and v not in depots_set)
            ]
        # Shallow copy with replaced edges; tours are left intact
        solution_for_overlay = {
            "depot": solution.get("depot"),
            "depots": solution.get("depots", []),
            "tours": solution["tours"],
            "edges": filtered_edges,
            "agent_depot": solution.get("agent_depot", {}),
        }

    # Overlay solution routes using the helper (now with arrows)
    plot_solution_overlay(
        ax=ax,
        coords=coords,
        solution=solution_for_overlay,
        show_depot=False,  # depots already plotted above
        arrows=True,
        arrowstyle="-|>",
        mutation_scale=12.0,
        arrow_every=1,
    )

    # Optional node annotations
    if annotate:
        for i, (x, y) in enumerate(xy.tolist()):
            ax.annotate(
                str(i), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8
            )

    ax.set_aspect("equal", adjustable="box")
    if N > 0:
        ax.set_xlim(float(xy[:, 0].min()) - 0.02, float(xy[:, 0].max()) + 0.02)
        ax.set_ylim(float(xy[:, 1].min()) - 0.02, float(xy[:, 1].max()) + 0.02)
    ax.grid(True, alpha=0.2)
    ax.set_title(f"{instance_name} — Solution")
    # ax.legend(loc="best")  # move legend below the figure

    # Place legend below the plot, close to the axes
    try:
        handles, labels = ax.get_legend_handles_labels()
        filtered = [
            (h, l) for h, l in zip(handles, labels) if l and not str(l).startswith("_")
        ]
        if filtered:
            handles, labels = zip(*filtered)
            ncol = max(1, min(4, len(labels)))
            fig = ax.figure
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),  # close to the plot
                ncol=ncol,
                borderaxespad=0.0,
                frameon=True,
            )
            fig.subplots_adjust(bottom=0.10)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    except Exception:
        pass

    plt.show()
