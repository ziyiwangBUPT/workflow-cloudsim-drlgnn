import tyro
import copy
import dataclasses
import random

import numpy as np

from gym_simulator.algorithms import algorithm_strategy
from gym_simulator.environments.static import StaticCloudSimEnvironment


@dataclasses.dataclass
class Args:
    simulator: str
    """path to the simulator JAR file"""
    host_count: int = 10
    """number of hosts"""
    vm_count: int = 10
    """number of VMs"""
    workflow_count: int = 5
    """number of workflows"""
    task_limit: int = 5
    """maximum number of tasks"""


def main(args: Args):
    env_config = {
        "host_count": args.host_count,
        "vm_count": args.vm_count,
        "workflow_count": args.workflow_count,
        "task_limit": args.task_limit,
        "simulator_mode": "embedded",
        "simulator_kwargs": {
            "simulator_jar_path": args.simulator,
            "verbose": False,
            "remote_debug": False,
        },
    }
    algorithms = [
        "round_robin",
        "max_min",
        "min_min",
        "best_fit",
        "fjssp_fifo_spt",
        "fjssp_fifo_eet",
        "fjssp_mopnr_spt",
        "fjssp_mopnr_eet",
        "fjssp_lwkr_spt",
        "fjssp_lwkr_eet",
        "fjssp_mwkr_spt",
        "fjssp_mwkr_eet",
    ]

    for algorithm in algorithms:
        random.seed(0)
        np.random.seed(0)

        env = StaticCloudSimEnvironment(env_config=copy.deepcopy(env_config))
        scheduler = algorithm_strategy.get_scheduler(algorithm)

        (tasks, vms), _ = env.reset()
        action = scheduler.schedule(tasks, vms)
        _, reward, terminated, truncated, _ = env.step(action)
        assert terminated or truncated, "Static environment should terminate after one step"

        print(f"Algorithm: {algorithm}, Reward: {reward}")

    env.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
