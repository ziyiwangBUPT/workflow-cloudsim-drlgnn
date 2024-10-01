import tyro
import dataclasses
import random

import numpy as np

from gym_simulator.environments.rl import RlCloudSimEnvironment


@dataclasses.dataclass
class Args:
    simulator: str
    """path to the simulator JAR file"""
    render_mode: str | None = "human"
    """render mode"""
    host_count: int = 10
    """number of hosts"""
    vm_count: int = 10
    """number of VMs"""
    workflow_count: int = 5
    """number of workflows"""
    task_limit: int = 5
    """maximum number of tasks"""
    algorithm: str = "round_robin"
    """algorithm to use"""
    remote_debug: bool = False
    """enable remote debugging"""


def main(args: Args):
    random.seed(0)
    np.random.seed(0)

    env = RlCloudSimEnvironment(
        env_config={
            "host_count": args.host_count,
            "vm_count": args.vm_count,
            "workflow_count": args.workflow_count,
            "task_limit": args.task_limit,
            "simulator_mode": "embedded",
            "simulator_kwargs": {
                "simulator_jar_path": args.simulator,
                "verbose": True,
                "remote_debug": args.remote_debug,
            },
            "render_mode": args.render_mode,
        },
    )

    actions = [
        (8, 8),
        (7, 6),
        (11, 8),
        (2, 0),
        (12, 9),
        (10, 8),
        (9, 4),
        (14, 8),
        (15, 6),
        (3, 0),
        (13, 9),
        (4, 8),
        (16, 0),
        (6, 4),
        (5, 1),
        (17, 8),
        (1, 0),
    ]

    obs, _ = env.reset()
    for task_id, vm_id in actions:
        action = {"vm_id": vm_id, "task_id": task_id}
        obs, reward, terminated, truncated, info = env.step(action)
        print("Reward:", reward)
        if terminated or truncated:
            print("Terminated")
            print(info)
            break

    env.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
