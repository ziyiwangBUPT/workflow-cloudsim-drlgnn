import click

from gym_simulator.releaser.environment import CloudSimReleaserEnvironment


@click.command()
@click.option("--simulator", help="Path to the simulator JAR file", required=True, type=click.Path(exists=True))
@click.option("--dataset", help="Path to the dataset JSON file", required=True, type=click.Path(exists=True))
@click.option("--render-mode", help="Render mode", type=click.Choice(["human"]))
def main(simulator: str, dataset: str, render_mode: str | None):
    env = CloudSimReleaserEnvironment(
        env_config={
            "simulator_mode": "embedded",
            "simulator_kwargs": {"simulator_jar_path": simulator, "dataset_path": dataset},
            "render_mode": render_mode,
        },
    )

    observation, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
