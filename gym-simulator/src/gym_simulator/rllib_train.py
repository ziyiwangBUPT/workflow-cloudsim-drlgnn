import tyro
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.air import RunConfig
from ray.rllib.algorithms.ppo import PPO

from gym_simulator.args import Args
from gym_simulator.releaser.environment import CloudSimReleaserEnvironment


def main(args: Args):
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
        param_space=config,
        tune_config=tune.TuneConfig(num_samples=1),
        run_config=RunConfig(storage_path=Path("./logs").absolute(), name="ppo-releaser"),
    )
    tuner.fit()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
