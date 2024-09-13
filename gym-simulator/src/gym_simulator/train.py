import click

from gym_simulator.releaser.env import CloudSimReleaserEnv
from gym_simulator.core.runner import CloudSimSimulatorRunner


@click.command()
@click.option("--simulator", help="Path to the simulator JAR file", required=True, type=click.Path(exists=True))
@click.option("--dataset", help="Path to the dataset JSON file", required=True, type=click.Path(exists=True))
def main(simulator: str, dataset: str):
    runner = CloudSimSimulatorRunner(simulator, dataset)
    env = CloudSimReleaserEnv(runner=runner, render_mode=None)

    try:
        obs, _ = env.reset()
        for _ in range(1000):
            action = 1 if obs[0] - obs[1] > 100 else 0
            print("Taking action:", action)
            obs, reward, terminated, truncated, info = env.step(action)
            print("Reward:", reward)
            if terminated or truncated:
                break
    except Exception as e:
        print("An error occurred:", e)
    finally:
        env.close()

    print("Exiting")


if __name__ == "__main__":
    main()
