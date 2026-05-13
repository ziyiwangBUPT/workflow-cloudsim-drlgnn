# # docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
from pathlib import Path
import random
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from tqdm import tqdm
import tyro
from torch.utils.tensorboard import SummaryWriter

from icecream import ic

from scheduler.config.settings import MIN_TESTING_DS_SEED
from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.rl_model.agents.agent import Agent
from scheduler.rl_model.agents.gin_agent.agent import GinAgent
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment


@dataclass
class Args:
    exp_name: str = "testnew1231"
    """the name of this experiment"""

    seed: int = 1
    """seed of the experiment"""
    output_dir: str = "logs"
    """the output directory of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    gpu_id: int = 2
    """which CUDA device index to use when cuda is enabled (e.g., 0, 1)"""
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
    test_iterations: int = 4
    """number of test iterations"""

    # Algorithm specific arguments
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 16
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
    target_kl: float | None = 0.02
    """the target KL divergence threshold"""

    dataset: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            host_count=4,
            vm_count=5,
            workflow_count=20,
            gnp_min_n=10,
            gnp_max_n=25,
            max_memory_gb=10,
            min_cpu_speed=1000,
            max_cpu_speed=5000,
            min_task_length=500000,
            max_task_length=100_0000,
            task_arrival="static",
            dag_method="gnp",
        )
    )
    """the dataset generation parameters"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    run_name: str = ""
    """the full name of the run"""


# Environment creation
# ----------------------------------------------------------------------------------------------------------------------


def make_env(idx: int, args: Args):
    assert not args.capture_video, "Video capturing is not yet supported"

    env: gym.Env = CloudSchedulingGymEnvironment(dataset_args=args.dataset)
    if args.capture_video and idx == 0:
        video_dir = f"{args.output_dir}/{args.run_name}/videos"
        env = RecordVideo(env, video_dir, episode_trigger=lambda x: x % 1000 == 0)
    env = GinAgentWrapper(env)
    return RecordEpisodeStatistics(env)


def make_test_env(args: Args):
    env: gym.Env = CloudSchedulingGymEnvironment(dataset_args=args.dataset)
    return GinAgentWrapper(env)


def make_agent(device: torch.device) -> Agent:
    return GinAgent(device)


# Training Agent
# ----------------------------------------------------------------------------------------------------------------------


def train(args: Args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.run_name = f"{int(time.time())}_{args.exp_name}"
    if args.track:
        import wandb

        assert args.wandb_project_name is not None, "Please specify the wandb project name"
        assert args.wandb_entity is not None, "Please specify the entity of wandb project"
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=args.run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"{args.output_dir}/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu_id)

    # env setup
    envs = gym.vector.SyncVectorEnv([lambda: make_env(i, args) for i in range(args.num_envs)])
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    assert isinstance(act_space, gym.spaces.Discrete), "only discrete action space is supported"
    assert obs_space.shape is not None
    assert act_space.shape is not None

    agent = make_agent(device)
    writer.add_text("agent", f"```{agent}```")

    last_model_save = 0
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
    # Use different seeds per parallel env to diversify datasets (must be python ints)
    per_env_seeds = [int(args.seed + i) for i in range(args.num_envs)]
    next_obs, _ = envs.reset(seed=per_env_seeds)
    next_obs_tensor = torch.Tensor(next_obs).to(device)
    next_done_tensor = torch.zeros(args.num_envs).to(device)

    pbar = tqdm(total=args.total_timesteps)
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

        with torch.no_grad():
            test_results = test_agent(agent, args)
            writer.add_scalar("tests/makespan", test_results[0], global_step)
            writer.add_scalar("tests/energy_consumption", test_results[1], global_step)
            writer.add_scalar("tests/carbon_emissions", test_results[2], global_step)

        if (global_step - last_model_save) >= 10_000:
            torch.save(agent.state_dict(), f"{args.output_dir}/{args.run_name}/model_{global_step}.pt")
            last_model_save = global_step

    torch.save(agent.state_dict(), f"{args.output_dir}/{args.run_name}/model.pt")

    envs.close()
    writer.close()

    pbar.close()


# Testing Agent
# ----------------------------------------------------------------------------------------------------------------------


def test_agent(agent: Agent, args: Args):
    total_makespan = 0.0
    total_energy_consumption = 0.0
    total_carbon_emissions = 0.0

    for seed_index in range(args.test_iterations):
        test_env = make_test_env(args)

        next_obs, _ = test_env.reset(seed=MIN_TESTING_DS_SEED + seed_index)
        while True:
            obs_tensor = torch.from_numpy(next_obs.astype(np.float32).reshape(1, -1))
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            vm_action = int(action.item())
            next_obs, _, terminated, truncated, _ = test_env.step(vm_action)
            if terminated or truncated:
                break

        assert test_env.prev_obs is not None
        total_makespan += test_env.prev_obs.makespan()
        total_energy_consumption += test_env.prev_obs.energy_consumption()
        # Sum carbon emissions across all tasks in the finished episode
        try:
            episode_carbon = 0.0
            for task in test_env.prev_obs.task_observations:
                # task.carbon_cost 是任务能耗乘以执行期间的平均碳强度
                episode_carbon += float(getattr(task, "carbon_cost", 0.0))
            total_carbon_emissions += max(0.0, episode_carbon)
        except Exception:
            # 如果不可用，则回退为0，不影响其他指标
            total_carbon_emissions += 0.0
        test_env.close()

    avg_makespan = total_makespan / args.test_iterations
    avg_energy_consumption = total_energy_consumption / args.test_iterations
    avg_carbon_emissions = total_carbon_emissions / args.test_iterations
    return avg_makespan, avg_energy_consumption, avg_carbon_emissions


if __name__ == "__main__":
    train(tyro.cli(Args))
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
# from pathlib import Path
# import random
# import time
# from dataclasses import dataclass, field
#
# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
# from sympy import false
# from tqdm import tqdm
# import tyro
# from torch.utils.tensorboard import SummaryWriter
#
# from icecream import ic
#
# from scheduler.config.settings import MIN_TESTING_DS_SEED
# from scheduler.dataset_generator.gen_dataset import DatasetArgs
# from scheduler.rl_model.agents.agent import Agent
# from scheduler.rl_model.agents.gin_agent.agent import GinAgent
# from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
# from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
# from scheduler.config.settings import DATA_PATH
#
#
# @dataclass
# class Args:
#     exp_name: str = "test"
#     """the name of this experiment"""
#
#     seed: int = 1
#     """seed of the experiment"""
#     output_dir: str = "logs"
#     """the output directory of the experiment"""
#     torch_deterministic: bool = True
#     """if toggled, `torch.backends.cudnn.deterministic=False`"""
#     cuda: bool = True
#     """if toggled, cuda will be enabled by default"""
#     gpu_id: int = 1
#     """which CUDA device index to use when cuda is enabled (e.g., 0, 1)"""
#     track: bool = False
#     """if toggled, this experiment will be tracked with Weights and Biases"""
#     wandb_project_name: str | None = None
#     """the wandb's project name"""
#     wandb_entity: str | None = None
#     """the entity (team) of wandb's project"""
#     capture_video: bool = False
#     """whether to capture videos of the agent performances (check out `videos` folder)"""
#     load_model_dir: str | None = None
#     """Directory to load the model from"""
#     test_iterations: int = 4
#     """number of test iterations"""
#
#     # Algorithm specific arguments
#     total_timesteps: int = 1000000
#     """total timesteps of the experiments"""
#     learning_rate: float = 3e-4
#     """the learning rate of the optimizer"""
#     num_envs: int = 32
#     """the number of parallel game environments"""
#     num_steps: int = 256
#     """the number of steps to run in each environment per policy rollout"""
#     anneal_lr: bool = false
#     """Toggle learning rate annealing for policy and value networks"""
#     gamma: float = 0.99
#     """the discount factor gamma"""
#     gae_lambda: float = 0.95
#     """the lambda for the general advantage estimation"""
#     num_minibatches: int = 16
#     """the number of mini-batches"""
#     update_epochs: int = 4
#     """the K epochs to update the policy"""
#     norm_adv: bool = True
#     """Toggles advantages normalization"""
#     clip_coef: float = 0.2
#     """the surrogate clipping coefficient"""
#     clip_vloss: bool = True
#     """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
#     ent_coef: float = 0.01
#     """coefficient of the entropy"""
#     vf_coef: float = 0.5
#     """coefficient of the value function"""
#     max_grad_norm: float = 0.5
#     """the maximum norm for the gradient clipping"""
#     target_kl: float | None = 0.02
#     """the target KL divergence threshold"""
#
#     dataset: DatasetArgs = field(
#         default_factory=lambda: DatasetArgs(
#             host_count=4,
#             vm_count=6,
#             workflow_count=10,
#             gnp_min_n=10,
#             gnp_max_n=20,
#             max_memory_gb=10,
#             min_cpu_speed=500,
#             max_cpu_speed=5000,
#             min_task_length=5000,
#             max_task_length=100_000,
#             task_arrival="static",
#             dag_method="gnp",
#         )
#     )
#     """the dataset generation parameters"""
#
#     # to be filled in runtime
#     batch_size: int = 0
#     """the batch size (computed in runtime)"""
#     minibatch_size: int = 0
#     """the mini-batch size (computed in runtime)"""
#     num_iterations: int = 0
#     """the number of iterations (computed in runtime)"""
#     run_name: str = ""
#     """the full name of the run"""
#
#
# # Environment creation
# # ----------------------------------------------------------------------------------------------------------------------
#
#
# def make_env(idx: int, args: Args):
#     assert not args.capture_video, "Video capturing is not yet supported"
#
#     # Sample a dataset configuration to improve generalization:
#     # randomly choose between GNP or Pegasus (Montage/Genome/Inspiral)
#     ds = sample_dataset_args(args.dataset)
#     env: gym.Env = CloudSchedulingGymEnvironment(dataset_args=ds)
#     if args.capture_video and idx == 0:
#         video_dir = f"{args.output_dir}/{args.run_name}/videos"
#         env = RecordVideo(env, video_dir, episode_trigger=lambda x: x % 1000 == 0)
#     env = GinAgentWrapper(env)
#     return RecordEpisodeStatistics(env)
#
#
# def make_test_env(args: Args):
#     # Use the same randomization strategy for test envs to evaluate generalization
#     ds = sample_dataset_args(args.dataset)
#     env: gym.Env = CloudSchedulingGymEnvironment(dataset_args=ds)
#     return GinAgentWrapper(env)
#
#
# def sample_dataset_args(base: DatasetArgs) -> DatasetArgs:
#     """
#     Create a randomized copy of DatasetArgs to improve training-time diversity.
#     Randomly choose among:
#       - gnp (keep base task sizes)
#       - pegasus: Montage (workflow_count=10), Genome (workflow_count=8), Inspiral (workflow_count=2)
#     Note: Pegasus generator will use global WORKFLOW_FILES; we vary only dag_method and workflow_count here.
#     """
#     choice = random.choice(["gnp", "pegasus_montage", "pegasus_genome", "pegasus_inspiral"])
#
#     if choice == "gnp":
#         # Slightly vary gnp task sizes to avoid overfitting to one n
#         vary = random.choice([0, 5, 10])
#         gnp_min_n = max(5, base.gnp_min_n + vary) if hasattr(base, "gnp_min_n") else 10
#         gnp_max_n = max(gnp_min_n, base.gnp_max_n + vary) if hasattr(base, "gnp_max_n") else gnp_min_n
#         return DatasetArgs(
#             host_count=base.host_count,
#             vm_count=base.vm_count,
#             workflow_count=base.workflow_count,
#             gnp_min_n=gnp_min_n,
#             gnp_max_n=gnp_max_n,
#             max_memory_gb=base.max_memory_gb,
#             min_cpu_speed=base.min_cpu_speed,
#             max_cpu_speed=base.max_cpu_speed,
#             min_task_length=base.min_task_length,
#             max_task_length=base.max_task_length,
#             task_arrival=base.task_arrival,
#             dag_method="gnp",
#         )
#
#     # Pegasus variants
#     if choice == "pegasus_montage":
#         wf_count = 13
#     elif choice == "pegasus_genome":
#         wf_count = 1
#     else:  # "pegasus_inspiral"
#         wf_count = 10
#
#     return DatasetArgs(
#         host_count=base.host_count,
#         vm_count=base.vm_count,
#         workflow_count=wf_count,
#         # gnp fields are ignored in pegasus mode, keep minimal values
#         gnp_min_n=1,
#         gnp_max_n=1,
#         max_memory_gb=base.max_memory_gb,
#         min_cpu_speed=base.min_cpu_speed,
#         max_cpu_speed=base.max_cpu_speed,
#         min_task_length=base.min_task_length,
#         max_task_length=base.max_task_length,
#         task_arrival=base.task_arrival,
#         dag_method="pegasus",
#     )
#
#
# def make_agent(device: torch.device) -> Agent:
#     return GinAgent(device)
#
#
# # Training Agent
# # ----------------------------------------------------------------------------------------------------------------------
#
#
# def train(args: Args):
#     args.batch_size = int(args.num_envs * args.num_steps)
#     args.minibatch_size = int(args.batch_size // args.num_minibatches)
#     args.num_iterations = args.total_timesteps // args.batch_size
#     args.run_name = f"{int(time.time())}_{args.exp_name}"
#     if args.track:
#         import wandb
#
#         assert args.wandb_project_name is not None, "Please specify the wandb project name"
#         assert args.wandb_entity is not None, "Please specify the entity of wandb project"
#         wandb.init(
#             project=args.wandb_project_name,
#             entity=args.wandb_entity,
#             sync_tensorboard=True,
#             config=vars(args),
#             name=args.run_name,
#             monitor_gym=True,
#             save_code=True,
#         )
#
#     writer = SummaryWriter(f"{args.output_dir}/{args.run_name}")
#     writer.add_text(
#         "hyperparameters",
#         "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
#     )
#
#     # TRY NOT TO MODIFY: seeding
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.backends.cudnn.deterministic = args.torch_deterministic
#
#     device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.cuda else "cpu")
#     if device.type == "cuda":
#         torch.cuda.set_device(args.gpu_id)
#
#     # env setup
#     envs = gym.vector.SyncVectorEnv([lambda: make_env(i, args) for i in range(args.num_envs)])
#     obs_space = envs.single_observation_space
#     act_space = envs.single_action_space
#     assert isinstance(act_space, gym.spaces.Discrete), "only discrete action space is supported"
#     assert obs_space.shape is not None
#     assert act_space.shape is not None
#
#     agent = make_agent(device)
#     writer.add_text("agent", f"```{agent}```")
#
#     last_model_save = 0
#     if args.load_model_dir:
#         model_path = Path(__file__).parent.parent.parent / "logs" / args.load_model_dir / "model.pt"
#         agent.load_state_dict(torch.load(str(model_path), weights_only=True))
#         print(f"Loaded model from {model_path}")
#
#     ic(agent)
#     optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
#
#     # ALGO Logic: Storage setup
#     obs = torch.zeros((args.num_steps, args.num_envs) + obs_space.shape).to(device)
#     actions = torch.zeros((args.num_steps, args.num_envs) + act_space.shape).to(device)
#     logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
#     rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
#     dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
#     values = torch.zeros((args.num_steps, args.num_envs)).to(device)
#
#     # TRY NOT TO MODIFY: start the game
#     global_step = 0
#     start_time = time.time()
#     # Use different seeds per parallel env to diversify datasets (must be python ints)
#     per_env_seeds = [int(args.seed + i) for i in range(args.num_envs)]
#     next_obs, _ = envs.reset(seed=per_env_seeds)
#     next_obs_tensor = torch.Tensor(next_obs).to(device)
#     next_done_tensor = torch.zeros(args.num_envs).to(device)
#
#     pbar = tqdm(total=args.total_timesteps)
#     for iteration in range(1, args.num_iterations + 1):
#         # Annealing the rate if instructed to do so.
#         if args.anneal_lr:
#             frac = 1.0 - (iteration - 1.0) / args.num_iterations
#             lrnow = frac * args.learning_rate
#             optimizer.param_groups[0]["lr"] = lrnow
#
#         for step in range(0, args.num_steps):
#             global_step += args.num_envs
#             obs[step] = next_obs_tensor
#             dones[step] = next_done_tensor
#
#             # ALGO LOGIC: action logic
#             with torch.no_grad():
#                 action, logprob, _, value = agent.get_action_and_value(next_obs_tensor)
#                 values[step] = value.flatten()
#             actions[step] = action
#             logprobs[step] = logprob
#
#             # TRY NOT TO MODIFY: execute the game and log data.
#             next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
#             next_done = np.logical_or(terminations, truncations)
#             rewards[step] = torch.Tensor(reward).to(device).view(-1)
#             next_obs_tensor, next_done_tensor = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
#
#             if "final_info" in infos:
#                 for info in infos["final_info"]:
#                     if info and "episode" in info:
#                         pbar.update(global_step - pbar.n)
#                         writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
#                         writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
#
#         # bootstrap value if not done
#         with torch.no_grad():
#             next_value = agent.get_value(next_obs_tensor).reshape(1, -1)
#             advantages = torch.zeros_like(rewards).to(device)
#             lastgaelam = 0
#             for t in reversed(range(args.num_steps)):
#                 if t == args.num_steps - 1:
#                     nextnonterminal = 1.0 - next_done_tensor
#                     nextvalues = next_value
#                 else:
#                     nextnonterminal = 1.0 - dones[t + 1]
#                     nextvalues = values[t + 1]
#                 delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
#                 advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
#             returns = advantages + values
#
#         # flatten the batch
#         b_obs = obs.reshape((-1,) + obs_space.shape)
#         b_logprobs = logprobs.reshape(-1)
#         b_actions = actions.reshape((-1,) + act_space.shape)
#         b_advantages = advantages.reshape(-1)
#         b_returns = returns.reshape(-1)
#         b_values = values.reshape(-1)
#
#         # Optimizing the policy and value network
#         b_inds = np.arange(args.batch_size)
#         clipfracs = []
#         for epoch in range(args.update_epochs):
#             np.random.shuffle(b_inds)
#             for start in range(0, args.batch_size, args.minibatch_size):
#                 end = start + args.minibatch_size
#                 mb_inds = b_inds[start:end]
#
#                 _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
#                 logratio = newlogprob - b_logprobs[mb_inds]
#                 ratio = logratio.exp()
#
#                 with torch.no_grad():
#                     # calculate approx_kl http://joschu.net/blog/kl-approx.html
#                     old_approx_kl = (-logratio).mean()
#                     approx_kl = ((ratio - 1) - logratio).mean()
#                     clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
#
#                 mb_advantages = b_advantages[mb_inds]
#                 if args.norm_adv:
#                     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
#
#                 # Policy loss
#                 pg_loss1 = -mb_advantages * ratio
#                 pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
#                 pg_loss = torch.max(pg_loss1, pg_loss2).mean()
#
#                 # Value loss
#                 newvalue = newvalue.view(-1)
#                 if args.clip_vloss:
#                     v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
#                     v_clipped = b_values[mb_inds] + torch.clamp(
#                         newvalue - b_values[mb_inds],
#                         -args.clip_coef,
#                         args.clip_coef,
#                     )
#                     v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
#                     v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
#                     v_loss = 0.5 * v_loss_max.mean()
#                 else:
#                     v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
#
#                 entropy_loss = entropy.mean()
#                 loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
#                 optimizer.step()
#
#             if args.target_kl is not None and approx_kl > args.target_kl:
#                 break
#
#         y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
#         var_y = np.var(y_true)
#         explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
#
#         # TRY NOT TO MODIFY: record rewards for plotting purposes
#         writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
#         writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
#         writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
#         writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
#         writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
#         writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
#         writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
#         writer.add_scalar("losses/explained_variance", explained_var, global_step)
#         writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
#
#         with torch.no_grad():
#             test_results = test_agent(agent, args)
#             writer.add_scalar("tests/makespan", test_results[0], global_step)
#             writer.add_scalar("tests/energy_consumption", test_results[1], global_step)
#             writer.add_scalar("tests/carbon_emissions", test_results[2], global_step)
#
#         if (global_step - last_model_save) >= 10_000:
#             torch.save(agent.state_dict(), f"{args.output_dir}/{args.run_name}/model_{global_step}.pt")
#             last_model_save = global_step
#
#     torch.save(agent.state_dict(), f"{args.output_dir}/{args.run_name}/model.pt")
#
#     envs.close()
#     writer.close()
#
#     pbar.close()
#
#
# # Testing Agent
# # ----------------------------------------------------------------------------------------------------------------------
#
#
# def test_agent(agent: Agent, args: Args):
#     total_makespan = 0.0
#     total_energy_consumption = 0.0
#     total_carbon_emissions = 0.0
#
#     for seed_index in range(args.test_iterations):
#         test_env = make_test_env(args)
#
#         next_obs, _ = test_env.reset(seed=MIN_TESTING_DS_SEED + seed_index)
#         while True:
#             obs_tensor = torch.from_numpy(next_obs.astype(np.float32).reshape(1, -1))
#             action, _, _, _ = agent.get_action_and_value(obs_tensor)
#             vm_action = int(action.item())
#             next_obs, _, terminated, truncated, _ = test_env.step(vm_action)
#             if terminated or truncated:
#                 break
#
#         assert test_env.prev_obs is not None
#         total_makespan += test_env.prev_obs.makespan()
#         total_energy_consumption += test_env.prev_obs.energy_consumption()
#         # Sum carbon emissions across all tasks in the finished episode
#         try:
#             episode_carbon = 0.0
#             for task in test_env.prev_obs.task_observations:
#                 # task.carbon_cost 是任务能耗乘以执行期间的平均碳强度
#                 episode_carbon += float(getattr(task, "carbon_cost", 0.0))
#             total_carbon_emissions += max(0.0, episode_carbon)
#         except Exception:
#             # 如果不可用，则回退为0，不影响其他指标
#             total_carbon_emissions += 0.0
#         test_env.close()
#
#     avg_makespan = total_makespan / args.test_iterations
#     avg_energy_consumption = total_energy_consumption / args.test_iterations
#     avg_carbon_emissions = total_carbon_emissions / args.test_iterations
#     return avg_makespan, avg_energy_consumption, avg_carbon_emissions
#
#
# if __name__ == "__main__":
#     train(tyro.cli(Args))
