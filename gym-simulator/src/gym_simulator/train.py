import click

from pprint import pprint
from ray.rllib.algorithms.ppo import PPOConfig

from gym_simulator.releaser.environment import CloudSimReleaserEnvironment


@click.command()
@click.option("--simulator", help="Path to the simulator JAR file", required=True, type=click.Path(exists=True))
@click.option("--dataset", help="Path to the dataset JSON file", required=True, type=click.Path(exists=True))
def main(simulator: str, dataset: str):
    config = (
        PPOConfig()
        .framework("torch")
        .env_runners(num_env_runners=1)
        .environment(
            CloudSimReleaserEnvironment,
            env_config={
                "simulator_mode": "embedded",
                "simulator_kwargs": {"simulator_jar_path": simulator, "dataset_path": dataset},
                "render_mode": None,
            },
        )
    )
    config.sample_timeout_s = 300
    algo = config.build()

    for i in range(5):
        results = algo.train()
        results.pop("config")
        pprint(results)


if __name__ == "__main__":
    main()
