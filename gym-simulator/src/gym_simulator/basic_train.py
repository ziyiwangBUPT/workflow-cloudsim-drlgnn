import tyro

from gym_simulator.args import TrainArgs
from gym_simulator.releaser.environment import CloudSimReleaserEnvironment


def main(args: TrainArgs):
    env = CloudSimReleaserEnvironment(
        env_config={
            "simulator_mode": "embedded",
            "simulator_kwargs": {"simulator_jar_path": args.simulator, "dataset_path": args.dataset},
            "render_mode": args.render_mode,
        },
    )

    observation, info = env.reset()
    total_reward: float = 0
    t = 0
    for _ in range(1000):
        action = int((observation[0] - observation[1]) >= 100)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        t += 1
        if terminated or truncated:
            break

    print(f"Total reward: {total_reward}")
    print(f"Total steps: {t}")
    env.close()


if __name__ == "__main__":
    args = tyro.cli(TrainArgs)
    main(args)
