"""
Training script for AM model

* [Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475)

* [POMO: Policy Optimization with Multiple Optima for Reinforcement Learning](https://arxiv.org/abs/2010.16011)

"""

import sys

sys.path.insert(0, "../")

import argparse

from tqdm import trange
from itertools import chain
import yaml

import time
import random
import os.path as osp
import os
import torch

import wandb

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

import importlib

from maenvs4vrp.learning.AM.policy_net_am_net_sing_ag import PolicyNet


def save_model_state_dict(save_path, model_policy):
    # save the policy state dict
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state_dict = model_policy.to("cpu").state_dict()
    torch.save(state_dict, save_path)


def set_random_seed(seed, torch_deterministic):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def reinforce_pomo_loss(logprobs, rewards, entropy, step_mask, args, reduction="mean"):
    r"""
    :param logprobs:  Iterable of length :math:`L` on tensors of size :math:`N \times 1`
    :param rewards:   Iterable of length :math:`L` on tensors of size :math:`N \times 1`
                    or single tensor of size :math:`N \times 1` to use rewards cumulated on the whole trajectory
    :param reduction: 'none' No reduction,
                      'sum'  Compute sum of loss on batch,
                      'mean' Compute mean of loss on batch
    """

    logprobs = torch.stack(logprobs)
    rewards = torch.stack(rewards).squeeze(-1)
    entropy = torch.stack(entropy)
    step_mask = torch.stack(step_mask)

    final_reward = rewards.detach().sum(0)

    # compute advantages
    av_final_reward = (
        final_reward.reshape(args.n_augment, args.batch_size // args.n_augment)
        .mean(dim=0)
        .repeat(args.n_augment)
        .detach()
    )
    advantages = final_reward - av_final_reward  # advantage

    # normalize advantaegs
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    loss = -(advantages * logprobs + args.ent_coef * entropy) * step_mask
    loss = loss.sum(0)

    if reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    else:  # reduction == 'mean'
        return loss.mean()


def train_epoch(args, env, policy, optim, ep, writer):
    policy.train()

    ep_loss = 0
    ep_prob = 0
    ep_rew = 0
    ep_norm = 0
    ep_nagent = 0
    ep_nvnodes = 0

    with trange(
        args.iter_count, desc="Ep.#{: >3d}/{: <3d}".format(ep + 1, args.epoch_count)
    ) as progress:
        for episode in progress:
            td = env.reset_agent_select_observe(
                num_agents=args.num_agents,
                num_nodes=args.num_nodes,
                seed=args.seed + ep * args.iter_count + episode,
                force_visit=args.force_visit,
                sample_type="augment",
                n_augment=args.n_augment,
                obs_list=["agent_cur_node_idx", "nodes_static", "action_mask", "agent"],
            )

            logps = []
            step_mask = []
            entropy = []
            rewards = []

            node_stat_obs = td["observations"]["nodes_static_obs"]

            policy.make_cache_(nodes_obs=node_stat_obs)

            while not td["done"].all():
                # rollover the observation
                try:
                    node_dyn_obs = td["observations"]["node_dynamic_obs"]
                except Exception:
                    node_dyn_obs = None
                try:
                    action_mask = td["observations"]["action_mask"]
                except Exception:
                    action_mask = None
                try:
                    self_obs = td["observations"]["agent_obs"]
                except Exception:
                    self_obs = None
                try:
                    global_obs = td["observations"]["global_obs"]
                except Exception:
                    global_obs = None
                cur_node_idx = td["observations"]["agent_cur_node_idx"]

                # get action from the policy
                action, logprobs, ent = policy.get_action_and_logs(
                    nodes_obs=node_dyn_obs,
                    self_obs=self_obs,
                    global_obs=global_obs,
                    cur_node_idx=cur_node_idx,
                    action_mask=action_mask,
                )

                td["next_action"] = action.unsqueeze(1)
                # execute the environment and log data
                td = env.step_agent_select_observe(
                    td,
                    obs_list=[
                        "agent_cur_node_idx",
                        "nodes_static",
                        "action_mask",
                        "agent",
                    ],
                )

                logps.append(logprobs)
                entropy.append(ent)
                rewards.append(td["reward"] + td["penalty"])
                step_mask.append(~td["done"])

            # --------------------------------------------------------
            loss = reinforce_pomo_loss(logps, rewards, entropy, step_mask, args)

            prob = torch.stack(logps).sum(0).exp().mean()
            rew = torch.stack(rewards).sum(0).mean()

            optim.zero_grad()
            loss.backward()
            if args.max_grad_norm is not None:
                grad_norm = nn.utils.clip_grad_norm_(
                    chain.from_iterable(grp["params"] for grp in optim.param_groups),
                    args.max_grad_norm,
                )
            optim.step()

            progress.set_postfix_str(
                "l={:.4g} p={:9.4g} rew={:6.4g} |g|={:.4g}".format(
                    loss, prob, rew, grad_norm
                )
            )

            not_visited_nodes = (
                env.td_state["nodes"]["active_nodes_mask"].sum(-1).float() - 1
            )
            number_used_agents = (
                env.td_state["agents"]["visited_nodes"].sum(-1).gt(1).sum(-1).float()
            )

            writer.add_scalar(
                "losses/episodic_loss", loss.item(), ep * args.iter_count + episode
            )
            writer.add_scalar(
                "losses/episodic_prob", prob.item(), ep * args.iter_count + episode
            )
            writer.add_scalar(
                "train/episodic_return", rew.item(), ep * args.iter_count + episode
            )
            writer.add_scalar(
                "train/episodic_not_visited_nodes",
                not_visited_nodes.mean().item(),
                ep * args.iter_count + episode,
            )
            writer.add_scalar(
                "train/episodic_number_used_agents",
                number_used_agents.mean().item(),
                ep * args.iter_count + episode,
            )

            ep_loss += loss.item()
            ep_prob += prob.item()
            ep_rew += rew.item()
            ep_norm += grad_norm
            ep_nagent += number_used_agents.mean().item()
            ep_nvnodes += not_visited_nodes.mean().item()

    writer.add_scalar("losses/epoch_loss", ep_loss / args.iter_count, ep)
    writer.add_scalar("losses/epoch_prob", ep_prob / args.iter_count, ep)
    writer.add_scalar("train/epoch_return", ep_rew / args.iter_count, ep)
    writer.add_scalar("train/epoch_not_visited_nodes", ep_nvnodes / args.iter_count, ep)
    writer.add_scalar("train/epoch_number_used_agents", ep_nagent / args.iter_count, ep)

    return tuple(stat / args.iter_count for stat in (ep_loss, ep_prob, ep_rew, ep_norm))


def test_epoch(args, test_env, policy, ep, writer):
    policy.eval()

    total_reward = []
    not_visited_nodes = []
    number_used_agents = []

    with torch.no_grad():
        for instance_name in test_env.inst_generator.set_of_instances:
            td = test_env.reset_agent_select_observe(
                num_agents=args.num_agents,
                num_nodes=args.num_nodes,
                force_visit=args.force_visit,
                sample_type="saved",
                instance_name=instance_name,
                seed=0,
                obs_list=["agent_cur_node_idx", "nodes_static", "action_mask", "agent"],
            )

            f_reward = []
            node_stat_obs = td["observations"]["nodes_static_obs"]
            policy.make_cache_(nodes_obs=node_stat_obs)

            while not td["done"].all():
                # rollover the observation
                try:
                    node_dyn_obs = td["observations"]["node_dynamic_obs"]
                except Exception:
                    node_dyn_obs = None
                try:
                    action_mask = td["observations"]["action_mask"]
                except Exception:
                    action_mask = None
                try:
                    self_obs = td["observations"]["agent_obs"]
                except Exception:
                    self_obs = None
                try:
                    global_obs = td["observations"]["global_obs"]
                except Exception:
                    global_obs = None
                cur_node_idx = td["observations"]["agent_cur_node_idx"]

                # get action from the agent
                action, _, _ = policy.get_action_and_logs(
                    nodes_obs=node_dyn_obs,
                    self_obs=self_obs,
                    global_obs=global_obs,
                    cur_node_idx=cur_node_idx,
                    action_mask=action_mask,
                    deterministic=True,
                )

                # execute the environment and log data
                td["next_action"] = action.unsqueeze(1)
                td = test_env.step_agent_select_observe(
                    td,
                    obs_list=[
                        "agent_cur_node_idx",
                        "nodes_static",
                        "action_mask",
                        "agent",
                    ],
                )

                f_reward.append(td["reward"] + td["penalty"])

            total_reward.append(torch.cat(f_reward, dim=1).sum(-1))
            not_visited_nodes.append(
                test_env.td_state["nodes"]["active_nodes_mask"].sum(-1).float() - 1
            )
            number_used_agents.append(
                test_env.td_state["agents"]["visited_nodes"]
                .sum(-1)
                .gt(1)
                .sum(-1)
                .float()
            )

        total_reward = torch.cat(total_reward).mean()
        not_visited_nodes = torch.cat(not_visited_nodes).mean()
        number_used_agents = torch.cat(number_used_agents).mean()

        writer.add_scalar("eval/episodic_return", total_reward, ep)
        writer.add_scalar("eval/episodic_not_visited_nodes", not_visited_nodes, ep)
        writer.add_scalar("eval/episodic_number_used_agents", number_used_agents, ep)

    print("Reward on test dataset: {:5.2f}".format(total_reward))
    return total_reward, not_visited_nodes, number_used_agents


def train(args, writer):
    """ENV SETUP"""
    # for CVRP
    if args.vrp_env == "cvrp":
        feature_list = yaml.safe_load(
            """
                                    nodes_static:
                                        x_coordinate:
                                            feat: x_coordinate
                                            norm:
                                        y_coordinate:
                                            feat: y_coordinate
                                            norm:
                                        demand:
                                            feat: demand
                                            norm:
                                        is_depot:
                                            feat: is_depot
                                            norm:
                                    agent:
                                        - remaining_capacity
                                                            """
        )
    # for TOP
    elif args.vrp_env == "top":
        feature_list = yaml.safe_load(
            """
                                    nodes_static:
                                        x_coordinate:
                                            feat: x_coordinate
                                            norm:
                                        y_coordinate:
                                            feat: y_coordinate
                                            norm:
                                        profits:
                                            feat: profits
                                            norm:
                                        is_depot:
                                            feat: is_depot
                                            norm:
                                    agent:
                                        - frac_current_profit
                                        - frac_current_time
                                                            """
        )
    else:
        raise Warning("define feature_list for this environment")

    if args.selection == "rand":
        env_agent_selector_module_name = (
            f"maenvs4vrp.environments.{args.vrp_env}.env_agent_selector"
        )
        env_agent_selector = importlib.import_module(
            env_agent_selector_module_name
        ).RandomSelector()
    elif args.selection == "single":
        env_agent_selector_module_name = (
            f"maenvs4vrp.environments.{args.vrp_env}.env_agent_selector"
        )
        env_agent_selector = importlib.import_module(
            env_agent_selector_module_name
        ).AgentSelector()
    elif args.selection == "stime":
        env_agent_selector_module_name = (
            f"maenvs4vrp.environments.{args.vrp_env}.env_agent_selector"
        )
        env_agent_selector = importlib.import_module(
            env_agent_selector_module_name
        ).SmallestTimeAgentSelector()

    observations_module_name = f"maenvs4vrp.environments.{args.vrp_env}.observations"
    observations = importlib.import_module(observations_module_name).Observations(
        feature_list
    )

    generator_module_name = (
        f"maenvs4vrp.environments.{args.vrp_env}.instances_generator"
    )
    generator = importlib.import_module(generator_module_name).InstanceGenerator(
        device=args.device
    )

    environment_module_name = f"maenvs4vrp.environments.{args.vrp_env}.env"
    environment_module = importlib.import_module(environment_module_name)

    env_agent_reward_module_name = (
        f"maenvs4vrp.environments.{args.vrp_env}.env_agent_reward"
    )
    reward_evaluator = importlib.import_module(
        env_agent_reward_module_name
    ).DenseReward()

    env = environment_module.Environment(
        instance_generator_object=generator,
        obs_builder_object=observations,
        agent_selector_object=env_agent_selector,
        reward_evaluator=reward_evaluator,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    if args.val_set == "None":
        eval_generator = importlib.import_module(
            generator_module_name
        ).InstanceGenerator(device=args.device)
    else:
        set_of_instances = set(
            generator.get_list_of_benchmark_instances()[args.val_set]["validation"]
        )
        eval_generator = importlib.import_module(
            generator_module_name
        ).InstanceGenerator(set_of_instances=set_of_instances, device=args.device)
        args.eval_batch_size = None

    eval_env = environment_module.Environment(
        instance_generator_object=eval_generator,
        obs_builder_object=observations,
        agent_selector_object=env_agent_selector,
        reward_evaluator=reward_evaluator,
        device=args.device,
        batch_size=args.eval_batch_size,
        seed=args.eval_seed,
    )

    nodes_static_obs_dim = env.obs_builder.get_nodes_static_feat_dim()
    nodes_dynamic_obs_dim = env.obs_builder.get_nodes_dynamic_feat_dim()
    agent_obs_dim = env.obs_builder.get_agent_feat_dim()
    agents_obs_dim = env.obs_builder.get_other_agents_feat_dim()
    global_obs_dim = env.obs_builder.get_global_feat_dim()

    """ ALGO LOGIC: EPISODE STORAGE"""
    start_time = time.time()

    print("Initializing attention model...")
    policy = PolicyNet(
        nodes_stat_obs_dim=nodes_static_obs_dim,
        nodes_dyn_obs_dim=nodes_dynamic_obs_dim,
        agent_obs_dim=agent_obs_dim,
        global_obs_dim=global_obs_dim,
        embed_dim=128,
    )

    policy.to(args.device)

    # OPTIMIZER AND LR SCHEDULER
    print("Initializing Adam optimizer...")

    lr_sched = None
    optim = Adam(
        [
            {
                "params": policy.parameters(),
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            }
        ]
    )
    scheduler = MultiStepLR(optim, milestones=args.milestones, gamma=0.1)

    best_lb_total_return = -10000000

    """ TRAINING LOGIC """
    print("Running...")
    train_stats = []
    test_stats = []
    for ep in range(0, args.epoch_count):
        train_stats.append(train_epoch(args, env, policy, optim, ep, writer))
        scheduler.step()
        test_metrics = test_epoch(args, eval_env, policy, ep, writer)
        test_stats.append(test_metrics)
        latest_episodic_return = test_metrics[0]

        writer.add_scalar("charts/SPS", int(ep / (time.time() - start_time)), ep)

        print("\n-------------------------------------------\n")
        print("Saving latest model...")
        save_model_state_dict(
            osp.join(args.log_path, "models/latest_model_" + args.run_name + ".zip"),
            policy,
        )
        policy.to(args.device)
        print("done")

        if latest_episodic_return > best_lb_total_return:
            print("Old best model: {: .2f}".format(best_lb_total_return))
            best_lb_total_return = latest_episodic_return
            print("New best model: {: .2f}".format(latest_episodic_return))
            print("Saving new best model")
            save_model_state_dict(
                osp.join(args.log_path, "models/best_model_" + args.run_name + ".zip"),
                policy,
            )
            policy.to(args.device)
            print("done")
        else:
            print("No improvement")
            print(f"Latest model: {latest_episodic_return}")
            print(f"Current best model: {best_lb_total_return}")

        writer.add_scalar("eval/best_model_lb_total_reward", best_lb_total_return, ep)

        print("\n-------------------------------------------\n")

    print("saving latest model")
    save_model_state_dict(
        osp.join(args.log_path, "models/latest_model_" + args.run_name + ".zip"), policy
    )
    policy.to(args.device)
    print("done")
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vrp_env",
        type=str,
        default="cvrp",
        help="select the vrp environment to train on",
    )
    parser.add_argument("--num_agents", type=int, default=10, help="number of agents")
    parser.add_argument("--num_nodes", type=int, default=51, help="number of nodes")
    parser.add_argument(
        "--selection",
        type=str,
        default="single",
        choices=["rand", "single", "stime"],
        help="next agent selection strategy",
    )
    parser.add_argument("--val_set", type=str, default="None", help="validation set")
    args = parser.parse_args()
    return args


def get_args():
    args = parse_args()
    args.model_name = "am_model"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.learning_rate = 0.0001
    args.weight_decay = 1e-6
    args.norm_adv = False
    args.force_visit = False
    args.ent_coef = 0.0
    args.milestones = [80, 95]
    args.epoch_count = 100
    args.iter_count = 2500
    args.batch_size = 512
    args.n_augment = 64
    args.eval_batch_size = 512
    args.max_grad_norm = 2
    args.torch_deterministic = True
    args.seed = 2297
    args.log_path = "runs"
    args.eval_seed = 0
    args.time = time.strftime("%Y_%m_%d_%Hh%Mm")
    args.run_name = f"{args.model_name}_{args.vrp_env}_{args.selection}_{args.num_nodes}n_{args.num_agents}a_{args.time}"
    args.debug = True
    return args


def main(args):
    print("Training with args", args)

    if args.seed != None:
        set_random_seed(args.seed, args.torch_deterministic)

    if not args.debug:
        wandb.init(
            project="your project",
            entity="your entity",
            sync_tensorboard=True,
            config=vars(args),
            monitor_gym=False,
            save_code=False,
        )

    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    train(args, writer)


if __name__ == "__main__":
    main(get_args())
