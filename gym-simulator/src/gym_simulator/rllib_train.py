import tyro

from pprint import pprint
from ray.rllib.algorithms.ppo import PPOConfig

from gym_simulator.args import Args
from gym_simulator.releaser.environment import CloudSimReleaserEnvironment


def main(args: Args):
    config = (
        PPOConfig()
        .framework("torch")
        .env_runners(num_env_runners=4)
        .evaluation(evaluation_num_env_runners=4)
        .environment(
            CloudSimReleaserEnvironment,
            env_config={
                "simulator_mode": "embedded",
                "simulator_kwargs": {"simulator_jar_path": args.simulator, "dataset_path": args.dataset},
                "render_mode": args.render_mode,
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
    args = tyro.cli(Args)
    main(args)
