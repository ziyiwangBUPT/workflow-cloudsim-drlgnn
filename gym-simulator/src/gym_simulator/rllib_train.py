import tyro
import torch
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.air import RunConfig, CheckpointConfig
from ray.rllib.algorithms.ppo import PPO

from gym_simulator.args import TrainArgs
from gym_simulator.releaser.environment import CloudSimReleaserEnvironment


def main(args: TrainArgs):
    config = (
        PPOConfig()
        .framework("torch")
        .env_runners(num_env_runners=2)
        .evaluation(evaluation_num_env_runners=2, evaluation_interval=30)
        .environment(
            CloudSimReleaserEnvironment,
            env_config={
                "simulator_mode": "embedded",
                "simulator_kwargs": {"simulator_jar_path": args.simulator, "dataset_path": args.dataset},
                "render_mode": args.render_mode,
            },
        )
    )

    tuner = tune.Tuner(
        PPO,
        param_space=config.to_dict(),
        tune_config=tune.TuneConfig(num_samples=1),
        run_config=RunConfig(
            name="ppo-releaser",
            storage_path=str(Path("./logs").absolute()),
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="mean_accuracy",
                checkpoint_score_order="max",
                checkpoint_frequency=10,
            ),
        ),
    )
    tuner.fit()


if __name__ == "__main__":
    args = tyro.cli(TrainArgs)
    main(args)
