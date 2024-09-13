import click

from gym_simulator.releaser.env import CloudSimReleaserEnv


@click.command()
@click.option("--simulator", help="Path to the simulator JAR file", required=True, type=click.Path(exists=True))
@click.option("--dataset", help="Path to the dataset JSON file", required=True, type=click.Path(exists=True))
def main(simulator: str, dataset: str):
    env = CloudSimReleaserEnv(simulator, dataset)
    obs, _ = env.reset()
    for _ in range(1000):
        action = 1 if obs[0] - obs[1] > 100 else 0
        print("Taking action:", action)
        obs, reward, terminated, truncated, info = env.step(action)
        print("Reward:", reward)
        if terminated or truncated:
            env.close()
            break


if __name__ == "__main__":
    main()
