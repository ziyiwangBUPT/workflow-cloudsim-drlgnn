import tyro

from gym_simulator.args import Args
from gym_simulator.releaser.environment import CloudSimReleaserEnvironment


def main(args: Args):
    env = CloudSimReleaserEnvironment(
        env_config={
            "simulator_mode": "embedded",
            "simulator_kwargs": {"simulator_jar_path": args.simulator, "dataset_path": args.dataset},
            "render_mode": args.render_mode,
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
    args = tyro.cli(Args)
    main(args)
