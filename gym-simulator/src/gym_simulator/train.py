import click

from pprint import pprint
from ray.rllib.algorithms.ppo import PPOConfig

from gym_simulator.releaser.environment import CloudSimReleaserEnvironment


@click.command()
@click.option("--simulator", help="Path to the simulator JAR file", required=True, type=click.Path(exists=True))
@click.option("--dataset", help="Path to the dataset JSON file", required=True, type=click.Path(exists=True))
@click.option("--render-mode", help="Render mode", type=click.Choice(["human"]))
def main(simulator: str, dataset: str, render_mode: str | None):
    config = (
        PPOConfig()
        .framework("torch")
        .env_runners(num_env_runners=1)
        .environment(
            CloudSimReleaserEnvironment,
            env_config={
                "simulator_mode": "embedded",
                "simulator_kwargs": {"simulator_jar_path": simulator, "dataset_path": dataset},
                "render_mode": render_mode,
            },
        )
    )
    config.sample_timeout_s = 300
    config.evaluation_interval = 5  # type: ignore
    algo = config.build()

    for i in range(5):
        result = algo.train()
        result.pop("config")
        pprint(result)
        evaluation = algo.evaluate()
        pprint(evaluation)


if __name__ == "__main__":
    main()
