import tyro
import json
import dataclasses
import random

import matplotlib.pyplot as plt
import numpy as np

from dataset_generator.core.models import Solution
from dataset_generator.visualizers.plotters import plot_gantt_chart
from gym_simulator.algorithms import algorithm_strategy
from gym_simulator.environments.static import StaticCloudSimEnvironment


@dataclasses.dataclass
class Args:
    simulator: str
    """path to the simulator JAR file"""
    render_mode: str | None = None
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

    env = StaticCloudSimEnvironment(
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

    # Choose the algorithm
    scheduler = algorithm_strategy.get_scheduler(args.algorithm)

    # Since this is static, the step will be only called once
    (tasks, vms), _ = env.reset()
    action = scheduler.schedule(tasks, vms)
    _, reward, terminated, truncated, info = env.step(action)
    assert terminated or truncated, "Static environment should terminate after one step"

    # Render the output if available
    print("Reward:", reward)

    # Plot the Gantt chart
    solution = info.get("solution")
    assert solution is not None and isinstance(solution, Solution), "Solution is not available"
    _, ax = plt.subplots()
    plot_gantt_chart(ax, solution.dataset.workflows, solution.dataset.vms, solution.vm_assignments, label=True)
    plt.title(f"Algorithm: {args.algorithm}")
    plt.show()

    env.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
