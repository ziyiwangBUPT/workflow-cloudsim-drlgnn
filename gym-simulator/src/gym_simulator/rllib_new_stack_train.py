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
        .environment(
            CloudSimReleaserEnvironment,
            env_config={
                "simulator_mode": "embedded",
                "simulator_kwargs": {"simulator_jar_path": simulator, "dataset_path": dataset},
                "render_mode": render_mode,
            },
        )
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .resources(num_cpus_for_main_process=1)
        .learners(num_learners=0, num_gpus_per_learner=0)
        .env_runners(num_env_runners=1)
        .training(model={"uses_new_env_runners": True})
    )
    algo = config.build()

    for i in range(5):
        results = algo.train()
        results.pop("config")
        pprint(results)
        algo.evaluate()


if __name__ == "__main__":
    main()
