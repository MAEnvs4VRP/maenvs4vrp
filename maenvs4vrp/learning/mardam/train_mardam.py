"""
Training script for mardam model

see: https://gitlab.inria.fr/gbono/mardam
and the paper: https://ieeexplore.ieee.org/abstract/document/9151266

"""

import sys

sys.path.insert(0, "../")

import argparse

from tqdm import trange
from itertools import chain

import time
import random
import os.path as osp
import os
import torch

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

import importlib

from maenvs4vrp.learning.mardam._learner import AttentionLearner
from maenvs4vrp.learning.mardam.layers import reinforce_loss
from maenvs4vrp.learning.mardam.baselines import CriticBaseline


def save_model_state_dict(save_path, model_policy):
    # save the policy state dict
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state_dict = model_policy.to("cpu").state_dict()
    torch.save(state_dict, save_path)


def set_random_seed(seed, torch_deterministic):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def train_epoch(args, env, bl_wrapped_learner, optim, ep, writer):
    bl_wrapped_learner.learner.train()

    ep_loss = 0
    ep_prob = 0
    ep_val = 0
    ep_bl = 0
    ep_norm = 0

    with trange(
        args.iter_count, desc="Ep.#{: >3d}/{: <3d}".format(ep + 1, args.epoch_count)
    ) as progress:
        for episode in progress:
            td = env.reset(
                num_agents=args.num_agents,
                num_nodes=args.num_nodes,
                seed=args.seed + ep * args.iter_count + episode,
                sample_type="random",
            )

            actions, logps, rewards, bl_vals, step_mask = bl_wrapped_learner(env, td)

            loss = reinforce_loss(logps, rewards, bl_vals)

            prob = torch.stack(logps).sum(0).exp().mean()
            val = torch.stack(rewards).sum(0).mean()
            bl = bl_vals[0].mean()

            optim.zero_grad()
            loss.backward()
            if args.max_grad_norm is not None:
                grad_norm = nn.utils.clip_grad_norm_(
                    chain.from_iterable(grp["params"] for grp in optim.param_groups),
                    args.max_grad_norm,
                )
            optim.step()

            progress.set_postfix_str(
                "l={:.4g} p={:9.4g} val={:6.4g} bl={:6.4g} |g|={:.4g}".format(
                    loss, prob, val, bl, grad_norm
                )
            )

            not_visited_nodes = (
                env.td_state["nodes"]["active_nodes_mask"].sum(-1).float() - 1
            )
            number_used_agents = (
                env.td_state["agents"]["visited_nodes"].sum(-1).gt(1).sum(-1).float()
            )

            writer.add_scalar("losses/episodic_loss", loss.item(), episode)
            writer.add_scalar("losses/episodic_prob", prob.item(), episode)

            writer.add_scalar("train/episodic_return", val.item(), episode)
            writer.add_scalar("train/episodic_baseline", bl.item(), episode)

            writer.add_scalar(
                "train/episodic_not_visited_nodes",
                not_visited_nodes.mean().item(),
                episode,
            )
            writer.add_scalar(
                "train/episodic_number_used_agents",
                number_used_agents.mean().item(),
                episode,
            )

            ep_loss += loss.item()
            ep_prob += prob.item()
            ep_val += val.item()
            ep_bl += bl.item()
            ep_norm += grad_norm

    writer.add_scalar("losses/epoch_loss", ep_loss / args.iter_count, ep)
    writer.add_scalar("losses/epoch_prob", ep_prob / args.iter_count, ep)

    writer.add_scalar("train/epoch_return", ep_val / args.iter_count, ep)
    writer.add_scalar("train/epoch_baseline", ep_bl / args.iter_count, ep)

    return tuple(
        stat / args.iter_count for stat in (ep_loss, ep_prob, ep_val, ep_bl, ep_norm)
    )


def test_epoch(args, test_env, learner, ep, writer):
    learner.eval()

    reward = []
    not_visited_nodes = []
    number_used_agents = []

    with torch.no_grad():
        for instance_name in test_env.inst_generator.set_of_instances:
            td = test_env.reset(
                num_agents=args.num_agents,
                num_nodes=args.num_nodes,
                sample_type="saved",
                instance_name=instance_name,
                seed=0,
            )

            _, _, rs = learner(test_env, td, deterministic=True)

            reward.append(torch.stack(rs).sum(dim=0).squeeze(-1))
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

        total_reward = torch.cat(reward).mean()
        not_visited_nodes = torch.cat(not_visited_nodes).mean()
        number_used_agents = torch.cat(number_used_agents).mean()

        writer.add_scalar("eval/episodic_return", total_reward, ep)
        writer.add_scalar("eval/episodic_not_visited_nodes", not_visited_nodes, ep)
        writer.add_scalar("eval/episodic_number_used_agents", number_used_agents, ep)

    print("Reward on test dataset: {:5.2f}".format(total_reward))
    return total_reward, not_visited_nodes, number_used_agents


def train(args, writer):
    """ENV SETUP"""

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
    observations = importlib.import_module(observations_module_name).Observations()

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
    # reward_evaluator = importlib.import_module(env_agent_reward_module_name).SparseReward()

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

    nodes_static_feat_dim = env.obs_builder.get_nodes_static_feat_dim()
    nodes_dynamic_feat_dim = env.obs_builder.get_nodes_dynamic_feat_dim()
    nodes_feat_dim = nodes_static_feat_dim + nodes_dynamic_feat_dim
    agent_feat_dim = env.obs_builder.get_agent_feat_dim()
    agents_feat_dim = env.obs_builder.get_other_agents_feat_dim()
    global_feat_dim = env.obs_builder.get_global_feat_dim()

    """ ALGO LOGIC: EPISODE STORAGE"""
    start_time = time.time()

    print("Initializing attention model...")
    learner = AttentionLearner(
        cust_feat_size=nodes_static_feat_dim, veh_state_size=agents_feat_dim
    )
    learner.to(args.device)
    # BASELINE
    print("Initializing critic baseline...")

    baseline = CriticBaseline(learner, args.num_nodes - 1)
    baseline.to(args.device)
    print("Done.")

    # OPTIMIZER AND LR SCHEDULER
    print("Initializing Adam optimizer...")

    lr_sched = None
    optim = Adam(
        [
            {"params": learner.parameters(), "lr": args.learning_rate},
            {"params": baseline.parameters(), "lr": args.critic_rate},
        ]
    )

    best_lb_total_return = -10000000

    """ TRAINING LOGIC """
    print("Running...")
    train_stats = []
    test_stats = []
    for ep in range(0, args.epoch_count):
        train_stats.append(train_epoch(args, env, baseline, optim, ep, writer))
        test_metrics = test_epoch(args, eval_env, learner, ep, writer)
        test_stats.append(test_metrics)
        latest_episodic_return = test_metrics[0]

        writer.add_scalar("charts/SPS", int(ep / (time.time() - start_time)), ep)

        print("\n-------------------------------------------\n")
        print("Saving latest model...")
        save_model_state_dict(
            osp.join(args.log_path, "models/latest_model_" + args.run_name + ".zip"),
            learner,
        )
        learner.to(args.device)
        print("done")

        if latest_episodic_return > best_lb_total_return:
            print("Old best model: {: .2f}".format(best_lb_total_return))
            best_lb_total_return = latest_episodic_return
            print("New best model: {: .2f}".format(latest_episodic_return))
            print("Saving new best model")
            save_model_state_dict(
                osp.join(args.log_path, "models/best_model_" + args.run_name + ".zip"),
                learner,
            )
            learner.to(args.device)
            print("done")
        else:
            print("No improvement")
            print(f"Latest model: {latest_episodic_return}")
            print(f"Current best model: {best_lb_total_return}")

        writer.add_scalar("eval/best_model_lb_total_reward", best_lb_total_return, ep)

        print("\n-------------------------------------------\n")

    print("saving latest model")
    save_model_state_dict(
        osp.join(args.log_path, "models/latest_model_" + args.run_name + ".zip"),
        learner,
    )
    learner.to(args.device)
    print("done")
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vrp_env",
        type=str,
        default="toptw",
        help="select the vrp environment to train on",
    )
    parser.add_argument("--num_agents", type=int, default=5, help="number of agents")
    parser.add_argument("--num_nodes", type=int, default=101, help="number of nodes")
    parser.add_argument(
        "--selection",
        type=str,
        default="rand",
        choices=["rand", "single", "stime"],
        help="next agent selection strategy",
    )
    parser.add_argument("--val_set", type=str, default="None", help="validation set")
    args = parser.parse_args()
    return args


def get_args():
    args = parse_args()
    args.model_name = "mardan"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.learning_rate = 0.0001
    args.critic_rate = 0.001
    args.epoch_count = 100
    args.iter_count = 2500
    args.batch_size = 512
    args.eval_batch_size = 512
    args.max_grad_norm = 2
    args.torch_deterministic = True
    args.seed = 2297
    args.log_path = "runs"
    args.eval_seed = 0
    args.time = time.strftime("%Y_%m_%d_%Hh%Mm")
    args.run_name = f"{args.model_name}_{args.vrp_env}_{args.selection}_{args.num_nodes}n_{args.num_agents}a_{args.time}"
    return args


def main(args):
    print("Training with args", args)

    if args.seed != None:
        set_random_seed(args.seed, args.torch_deterministic)

    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    train(args, writer)


if __name__ == "__main__":
    main(get_args())
