import click

from ray.rllib.algorithms.ppo import PPOConfig

from gym_simulator.releaser.env import CloudSimReleaserEnv
from gym_simulator.core.runner import CloudSimSimulatorRunner


@click.command()
@click.option("--simulator", help="Path to the simulator JAR file", required=True, type=click.Path(exists=True))
@click.option("--dataset", help="Path to the dataset JSON file", required=True, type=click.Path(exists=True))
def main(simulator: str, dataset: str):
    config = (
        PPOConfig()
        .environment(
            CloudSimReleaserEnv,
            env_config={"runner": CloudSimSimulatorRunner(simulator, dataset), "render_mode": "human"},
        )
        .framework("torch")
        .env_runners(num_env_runners=1)
    )
    algo = config.build()

    for i in range(5):
        results = algo.train()
        print(f"Iter: {i}; avg. return={results['env_runners']['episode_return_mean']}")


if __name__ == "__main__":
    main()
