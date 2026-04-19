"""
PPO implementation adapted from: https://pettingzoo.farama.org/tutorials/cleanrl/
and https://github.com/vwxyzjn/cleanrl

"""

import os
import sys

sys.path.insert(0, "../")

import argparse
from tqdm import tqdm

import numpy as np

import time
import random
import torch

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

# from ml_collections import config_dict
import importlib


from maenvs4vrp.learning.ppo.policy_net_ma_ac import PolicyNet


def save_model_state_dict(save_path, model_policy):
    # save the policy state dict
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state_dict = model_policy.to("cpu").state_dict()
    torch.save(state_dict, save_path)


def set_random_seed(seed, torch_deterministic):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def train(args, writer):
    """ENV SETUP"""

    num_agents = args.num_agents
    num_nodes = args.num_nodes
    num_steps = args.num_steps
    n_envs = args.batch_size

    env_agent_selector_module_name = (
        f"maenvs4vrp.environments.{args.vrp_env}.env_agent_selector"
    )
    env_agent_selector = importlib.import_module(
        env_agent_selector_module_name
    ).AgentSelector()

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
    ).SparseReward()

    env = environment_module.Environment(
        instance_generator_object=generator,
        obs_builder_object=observations,
        agent_selector_object=env_agent_selector,
        reward_evaluator=reward_evaluator,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    eval_env = environment_module.Environment(
        instance_generator_object=generator,
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

    policy = PolicyNet(
        nodes_feat_dim=nodes_feat_dim,
        agent_feat_dim=agent_feat_dim,
        agents_feat_dim=agents_feat_dim,
        global_feat_dim=global_feat_dim,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

    best_lb_total_return = -10000000

    """ TRAINING LOGIC """
    # train for n number of episodes
    pbar = tqdm(range(args.total_episodes))
    for episode in pbar:
        if args.anneal_lr:
            frac = 1.0 - (episode - 1.0) / args.total_episodes
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # collect an episode
        with torch.no_grad():
            # collect episodes
            td = env.reset(
                num_agents=num_agents,
                num_nodes=num_nodes,
                sample_type="random",
                seed=args.seed + episode,
            )

            rb_node_obs = torch.zeros(
                (num_steps, n_envs, num_nodes, nodes_feat_dim)
            ).to(args.device)
            rb_actions_mask = torch.zeros(
                (num_steps, n_envs, num_nodes), dtype=torch.bool
            ).to(args.device)
            rb_self_obs = torch.zeros((num_steps, n_envs, agent_feat_dim)).to(
                args.device
            )
            rb_global_obs = torch.zeros((num_steps, n_envs, global_feat_dim)).to(
                args.device
            )
            rb_step_mask = torch.zeros((num_steps, n_envs), dtype=torch.bool).to(
                args.device
            )
            rb_agents_obs = torch.zeros(
                (num_steps, n_envs, num_agents, agents_feat_dim)
            ).to(args.device)
            rb_agents_mask = torch.zeros(
                (num_steps, n_envs, num_agents), dtype=torch.bool
            ).to(args.device)

            rb_actions = torch.zeros((num_steps, n_envs), dtype=torch.long).to(
                args.device
            )
            rb_logprobs = torch.zeros((num_steps, n_envs)).to(args.device)
            rb_rewards = torch.zeros((num_steps, n_envs)).to(args.device)
            rb_entropy = torch.zeros((num_steps, n_envs)).to(args.device)

            rb_values = torch.zeros((num_steps, n_envs)).to(args.device)
            final_reward = torch.zeros(n_envs).to(args.device)
            step_mask = torch.ones(n_envs, dtype=torch.bool).to(args.device)

            node_stat_obs = td["observations"]["node_static_obs"]

            while not td["done"].all():
                # rollover the observation
                node_dyn_obs = td["observations"]["node_dynamic_obs"]
                action_mask = td["observations"]["action_mask"]
                self_obs = td["observations"]["agent_obs"]
                global_obs = td["observations"]["global_obs"]
                agents_mask = td["observations"]["agents_mask"]
                agents_obs = td["observations"]["other_agents_obs"]
                node_obs = torch.cat((node_stat_obs, node_dyn_obs), dim=2)

                # get action from the agent
                action, logprobs, entropy, values = policy.get_action_and_logs(
                    nodes_obs=node_obs,
                    self_obs=self_obs,
                    agents_obs=agents_obs,
                    agents_mask=agents_mask,
                    global_obs=global_obs,
                    action_mask=action_mask,
                )

                td["action"] = action.unsqueeze(1)

                # execute the environment and log data
                step_idx = env.env_nsteps
                td = env.step(td)

                rb_node_obs[step_idx] = node_obs
                rb_actions_mask[step_idx] = action_mask
                rb_self_obs[step_idx] = self_obs
                rb_agents_obs[step_idx] = agents_obs
                rb_agents_mask[step_idx] = agents_mask
                rb_global_obs[step_idx] = global_obs

                rb_step_mask[step_idx] = step_mask
                rb_rewards[step_idx] = td["reward"].squeeze(1) + td["penalty"].squeeze(
                    1
                )

                rb_actions[step_idx] = action.to(torch.long)
                rb_logprobs[step_idx] = logprobs
                rb_entropy[step_idx] = entropy
                rb_values[step_idx] = values.squeeze(1)
                step_mask = ~td["done"]

            final_reward = rb_rewards.detach().sum(0)
            not_visited_nodes = (
                env.td_state["nodes"]["active_nodes_mask"].sum(-1).float() - 1
            )
            number_used_agents = (
                env.td_state["agents"]["visited_nodes"].sum(-1).gt(1).sum(-1).float()
            )

        # compute advantages
        with torch.no_grad():
            if args.gae:
                gae = 0
                rb_advantages = torch.zeros_like(rb_rewards).to(args.device)
                for step in reversed(range(rb_rewards.shape[0] - 1)):
                    delta = (
                        rb_rewards[step]
                        + args.gamma * rb_values[step + 1] * rb_step_mask[step + 1]
                        - rb_values[step]
                    )
                    rb_advantages[step] = gae = (
                        delta
                        + args.gamma * args.gae_lambda * rb_step_mask[step + 1] * gae
                    )

                rb_returns = rb_advantages + rb_values
            else:
                rb_returns = torch.zeros_like(rb_rewards).to(args.device)
                for step in reversed(range(rb_rewards.shape[0] - 1)):
                    rb_returns[step] = (
                        rb_returns[step + 1] * args.gamma * rb_step_mask[step + 1]
                        + rb_rewards[step]
                    )
                rb_advantages = rb_returns - rb_values

        # convert our episodes to batch of individual transitions
        b_node_obs = rb_node_obs[rb_step_mask]
        b_actions_mask = rb_actions_mask[rb_step_mask]
        b_self_obs = rb_self_obs[rb_step_mask]
        b_agents_obs = rb_agents_obs[rb_step_mask]
        b_agents_mask = rb_agents_mask[rb_step_mask]
        b_global_obs = rb_global_obs[rb_step_mask]

        b_actions = rb_actions[rb_step_mask]
        b_logprobs = rb_logprobs[rb_step_mask]
        b_values = rb_values[rb_step_mask]
        b_returns = rb_returns[rb_step_mask]
        b_advantages = rb_advantages[rb_step_mask]

        # Optimizing the policy and value network
        n_samples = b_advantages.size(0)
        assert n_samples >= args.batch_size, (
            "PPO requires the number step samples "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(args.batch_size)
        )

        b_index = np.arange(n_samples)

        clip_fracs = []
        policy.train()
        for repeat in range(args.update_epochs):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, n_samples, args.batch_size):
                # select the indices we want to train on
                end = start + args.batch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = policy.get_action_and_logs(
                    nodes_obs=b_node_obs[batch_index],
                    self_obs=b_self_obs[batch_index],
                    agents_obs=b_agents_obs[batch_index],
                    agents_mask=b_agents_mask[batch_index],
                    global_obs=b_global_obs[batch_index],
                    action_mask=b_actions_mask[batch_index],
                    action=b_actions[batch_index],
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                advantages = b_advantages[batch_index]
                # normalize advantaegs
                if args.norm_adv:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                # Value loss
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], episode
        )
        writer.add_scalar("losses/loss", loss.item(), episode)
        writer.add_scalar("losses/value_loss", v_loss.item(), episode)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), episode)
        writer.add_scalar("losses/entropy", entropy_loss.item(), episode)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), episode)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), episode)
        writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), episode)
        writer.add_scalar("losses/explained_variance", explained_var, episode)
        # print("SPS:", int(episode / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(episode / (time.time() - start_time)), episode
        )

        # av_total_episodic_rew = torch.mean(total_episodic_rew).item()
        av_total_episodic_return = torch.mean(final_reward).item()
        av_not_visited_nodes = torch.mean(not_visited_nodes).item()
        av_number_used_agents = torch.mean(number_used_agents).item()

        # writer.add_scalar("charts/episodic_reward:", av_total_episodic_rew, episode)
        writer.add_scalar("charts/episodic_return:", av_total_episodic_return, episode)
        writer.add_scalar(
            "charts/episodic_not_visited_nodes:", av_not_visited_nodes, episode
        )
        writer.add_scalar(
            "charts/episodic_number_used_agents:", av_number_used_agents, episode
        )

        pbar.set_description(
            "Episodic Return: {: .2f}, Not visited nodes: {}, Used agents: {}, Policy Loss: {:3.3f}, Value Loss: {:3.3f}, loss: {:3.3f}".format(
                av_total_episodic_return,
                av_not_visited_nodes,
                av_number_used_agents,
                pg_loss.item(),
                v_loss.item(),
                loss,
            )
        )
        # print("\n-------------------------------------------\n")

        if episode % args.eval_num_print == 0:
            print("\n-------------------------------------------\n")

            # print ('Saving latest model...')
            # save_model_state_dict(osp.join(args.log_path, "models/latest_model.zip"), agent_policy)
            # print ('done')

            print("Running eval on validation set")
            latest_episodic_return, not_visited_nodes, number_used_agents = evaluate(
                args, writer, eval_env, policy
            )
            latest_not_visited_nodes = torch.mean(not_visited_nodes).item()
            latest_episodic_return = torch.mean(latest_episodic_return).item()
            latest_number_used_agents = torch.mean(number_used_agents).item()

            print(f"number not visited nodes: {latest_not_visited_nodes}")
            print(f"number of used agents: {latest_number_used_agents}")

            if latest_episodic_return > best_lb_total_return:
                print("Old best model: {: .2f}".format(best_lb_total_return))
                best_lb_total_return = latest_episodic_return
                print("New best model: {: .2f}".format(latest_episodic_return))
                # print ('Saving new best model')
                # save_model_state_dict(osp.join(args.log_path, "models/best_model.zip"), agent_policy)
                print("done")
            else:
                print("No improvement")
                print(f"Latest model: {latest_episodic_return}")
                print(f"Current best model: {best_lb_total_return}")

            # writer.add_scalar("eval/latest_model_lb_total_reward", latest_lb_total_rew, episode)
            writer.add_scalar(
                "eval/best_model_lb_total_reward", best_lb_total_return, episode
            )
            # writer.add_scalar("eval/episodic_reward:", latest_lb_total_rew, episode)
            writer.add_scalar("eval/episodic_return:", latest_episodic_return, episode)
            writer.add_scalar(
                "eval/episodic_not_visited_nodes:", latest_not_visited_nodes, episode
            )
            writer.add_scalar(
                "eval/episodic_number_used_agents:", latest_number_used_agents, episode
            )
            print("\n-------------------------------------------\n")

    # print ('saving model')
    # print (osp.join(args.log_path, "models/latest_model.zip"))
    # save_model_state_dict(osp.join(args.log_path, "models/latest_model.zip"), agent_policy)

    print("done")
    writer.close()


def evaluate(args, writer, eval_env, policy):
    policy.eval()

    total_reward = []
    not_visited_nodes = []
    number_used_agents = []

    # collect an episode
    with torch.no_grad():
        # collect observations
        td = eval_env.reset(
            num_agents=args.num_agents, num_nodes=args.num_nodes, seed=args.eval_seed
        )
        f_reward = []
        node_stat_obs = td["observations"]["node_static_obs"]

        while not td["done"].all():
            # rollover the observation
            node_dyn_obs = td["observations"]["node_dynamic_obs"]
            action_mask = td["observations"]["action_mask"]
            self_obs = td["observations"]["agent_obs"]
            global_obs = td["observations"]["global_obs"]
            agents_mask = td["observations"]["agents_mask"]
            agents_obs = td["observations"]["other_agents_obs"]

            node_obs = torch.cat((node_stat_obs, node_dyn_obs), dim=2)

            # get action from the agent
            action, _, _, _ = policy.get_action_and_logs(
                nodes_obs=node_obs,
                self_obs=self_obs,
                agents_obs=agents_obs,
                agents_mask=agents_mask,
                global_obs=global_obs,
                action_mask=action_mask,
                deterministic=True,
            )

            # execute the environment and log data
            td["action"] = action.unsqueeze(1)
            td = eval_env.step(td)

            f_reward.append(td["reward"] + td["penalty"])

        total_reward = torch.cat(f_reward, dim=1).sum(-1)
        not_visited_nodes = (
            eval_env.td_state["nodes"]["active_nodes_mask"].sum(-1).float() - 1
        )
        number_used_agents = (
            eval_env.td_state["agents"]["visited_nodes"].sum(-1).gt(1).sum(-1).float()
        )
    return total_reward, not_visited_nodes, number_used_agents


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vrp_env",
        type=str,
        default="toptw",
        help="select the vrp environment to train on",
    )
    parser.add_argument("--num_agents", type=int, default=5, help="number of agents")
    parser.add_argument("--num_nodes", type=int, default=21, help="number of nodes")
    args = parser.parse_args()
    return args


def get_args():
    args = parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.ent_coef = 0.01
    args.vf_coef = 0.5
    args.clip_coef = 0.05
    args.gae = True
    args.gamma = 0.99
    args.gae_lambda = 0.95
    args.batch_size = 512
    args.eval_batch_size = 512

    args.hidden_dim = 128
    args.n_envs = 128

    args.num_steps = args.num_nodes + args.num_agents + 1
    args.total_episodes = 2 * 2500 + 1

    args.learning_rate = 1e-4
    args.update_epochs = 2

    args.anneal_lr = False
    args.max_grad_norm = 10
    args.norm_adv = False
    args.torch_deterministic = True
    args.seed = 2297

    args.exp_name = "test"

    args.eval_num_episodes = 1
    args.eval_num_print = 2500
    args.eval_seed = 9875
    return args


def main(args):
    print("Training with args", args)

    if args.seed != None:
        set_random_seed(args.seed, args.torch_deterministic)

    run_name = f"{args.vrp_env}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    train(args, writer)


if __name__ == "__main__":
    # main(parse_args())
    main(get_args())
