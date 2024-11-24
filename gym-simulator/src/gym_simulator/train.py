# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import dataclasses
import os
from pathlib import Path
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import tyro
from torch.utils.tensorboard import SummaryWriter

from icecream import ic

from gym_simulator.algorithms.rl_agents.gin_agent import GinAgent
from gym_simulator.args import TESTING_DS_ARGS, TRAINING_DS_ARGS
from gym_simulator.environments.rl_gym import RlGymCloudSimEnvironment


training_ds_args = dataclasses.asdict(TRAINING_DS_ARGS)
testing_ds_args = dataclasses.asdict(TESTING_DS_ARGS)


@dataclass
class Args:
    exp_name: str
    """the name of this experiment"""

    seed: int = 1
    """seed of the experiment"""
    output_dir: str = "logs"
    """the output directory of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str | None = None
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    load_model_dir: str | None = None
    """Directory to load the model from"""

    # Algorithm specific arguments
    total_timesteps: int = 2_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(idx: int, args: Args, video_dir: str):
    def thunk():
        env_config = {"simulator_mode": "internal", "simulator_kwargs": {"dataset_args": training_ds_args}}
        if args.capture_video and idx == 0:
            env_config["render_mode"] = "rgb_array"
            base_env = RlGymCloudSimEnvironment(env_config=env_config)
            env = gym.wrappers.RecordVideo(base_env, video_dir, episode_trigger=lambda x: x % 1000 == 0)
            return gym.wrappers.RecordEpisodeStatistics(env)

        base_env = RlGymCloudSimEnvironment(env_config=env_config)
        return gym.wrappers.RecordEpisodeStatistics(base_env)

    return thunk


def test_agent(agent: GinAgent, test_env: RlGymCloudSimEnvironment, test_count=10):
    total_makespan = 0
    total_power_consumption = 0

    # Run a simple environment loop to test the model
    for i in range(test_count):
        next_obs, _ = test_env.reset(seed=i + 11**7)
        while True:
            obs_tensor = torch.Tensor(next_obs.reshape(1, -1)).to("cpu")
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            vm_action = action.cpu().numpy()[0]
            next_obs, _, terminated, truncated, _ = test_env.step(vm_action)
            if terminated or truncated:
                break
        total_makespan += test_env.state.task_completion_time[-1]
        total_power_consumption += test_env.state.task_power_consumptions.sum()

    return total_makespan / test_count, total_power_consumption / test_count


def main(args: Args):
    pbar = tqdm(total=args.total_timesteps)
    last_model_save = 0

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{int(time.time())}_{args.exp_name}"
    if args.track:
        import wandb

        assert args.wandb_project_name is not None, "Please specify the wandb project name"
        assert args.wandb_entity is not None, "Please specify the entity of wandb project"
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    ic(training_ds_args)
    ic(testing_ds_args)

    writer = SummaryWriter(f"{args.output_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_video_dir = f"{args.output_dir}/{run_name}/videos"
    envs = gym.vector.SyncVectorEnv([make_env(i, args, video_dir=env_video_dir) for i in range(args.num_envs)])
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    assert isinstance(act_space, gym.spaces.Discrete), "only discrete action space is supported"
    assert obs_space.shape is not None
    assert act_space.shape is not None

    # test env setup
    testing_env_config = {"simulator_mode": "internal", "simulator_kwargs": {"dataset_args": testing_ds_args}}
    test_env = RlGymCloudSimEnvironment(testing_env_config)

    agent = GinAgent(device=device).to(device)
    writer.add_text("agent", f"```{agent}```")

    if args.load_model_dir:
        model_path = Path(__file__).parent.parent.parent / "logs" / args.load_model_dir / "model.pt"
        agent.load_state_dict(torch.load(str(model_path), weights_only=True))
        print(f"Loaded model from {model_path}")

    ic(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + act_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs_tensor = torch.Tensor(next_obs).to(device)
    next_done_tensor = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs_tensor
            dones[step] = next_done_tensor

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs_tensor)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.Tensor(reward).to(device).view(-1)
            next_obs_tensor, next_done_tensor = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        pbar.update(global_step - pbar.n)
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs_tensor).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done_tensor
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + obs_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + act_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        test_results = test_agent(agent, test_env)
        writer.add_scalar("tests/makespan", test_results[0], global_step)
        writer.add_scalar("tests/energy_consumption", test_results[1], global_step)

        if (global_step - last_model_save) >= 10_000:
            torch.save(agent.state_dict(), f"{args.output_dir}/{run_name}/model_{global_step}.pt")
            last_model_save = global_step

    torch.save(agent.state_dict(), f"{args.output_dir}/{run_name}/model.pt")

    test_env.close()
    envs.close()
    writer.close()

    pbar.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
