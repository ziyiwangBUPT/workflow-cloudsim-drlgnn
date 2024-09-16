import tyro
from pathlib import Path

from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.algorithm import Algorithm

from gym_simulator.args import TestArgs
from gym_simulator.releaser.environment import CloudSimReleaserEnvironment


def main(args: TestArgs):
    path = Path(args.checkpoint_dir).absolute()
    tuner = tune.Tuner.restore(str(path), PPO)

    results = tuner.get_results()
    best_result = results.get_best_result()
    algo = Algorithm.from_checkpoint(best_result.checkpoint)

    env = CloudSimReleaserEnvironment(
        env_config={
            "simulator_mode": "embedded",
            "simulator_kwargs": {"simulator_jar_path": args.simulator, "dataset_path": args.dataset},
            "render_mode": args.render_mode,
        }
    )

    episode_return = 0
    terminated = truncated = False
    obs, _ = env.reset()
    while not terminated and not truncated:
        action = algo.compute_single_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward

    print(f"Reached episode return of {episode_return}.")


if __name__ == "__main__":
    args = tyro.cli(TestArgs)
    main(args)
