import tyro
import dataclasses
import random

import numpy as np

from gym_simulator.environments.rl import RlCloudSimEnvironment


@dataclasses.dataclass
class Args:
    simulator: str
    """path to the simulator JAR file"""
    seed: int = 0
    """random seed"""
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
    env = RlCloudSimEnvironment(
        env_config={
            "host_count": args.host_count,
            "vm_count": args.vm_count,
            "workflow_count": args.workflow_count,
            "task_limit": args.task_limit,
            "simulator_mode": "embedded",
            "simulator_kwargs": {
                "simulator_jar_path": args.simulator,
                "verbose": False,
                "remote_debug": args.remote_debug,
                "dataset_args": {"seed": args.seed},
            },
            "render_mode": args.render_mode,
        },
    )

    obs, _ = env.reset()
    while True:
        task_state_ready = obs["task_state_ready"]
        task_completion_time = obs["task_completion_time"]
        vm_completion_time = obs["vm_completion_time"]
        task_vm_compatibility = obs["task_vm_compatibility"]
        task_vm_time_cost = obs["task_vm_time_cost"]
        task_graph_edges = obs["task_graph_edges"]

        # Task ID is the ready task with the minimum completion time
        ready_task_ids = np.where(task_state_ready == 1)
        min_comp_time_of_ready_tasks = np.min(task_completion_time[ready_task_ids])
        next_task_id = np.where(task_completion_time == min_comp_time_of_ready_tasks)[0][0]

        # VM ID is the VM that can give the task the minimum completion time
        parent_ids = np.where(task_graph_edges[:, next_task_id] == 1)[0]
        max_parent_comp_time = max(task_completion_time[parent_ids])
        compatible_vm_ids = np.where(task_vm_compatibility[next_task_id] == 1)[0]
        min_comp_time = np.inf
        next_vm_id = None
        for vm_id in compatible_vm_ids:
            comp_time = max(max_parent_comp_time, vm_completion_time[vm_id]) + task_vm_time_cost[next_task_id, vm_id]
            if comp_time < min_comp_time:
                min_comp_time = comp_time
                next_vm_id = vm_id
        assert min_comp_time == min_comp_time_of_ready_tasks

        action = {"vm_id": next_vm_id, "task_id": next_task_id}
        print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        print("Reward:", reward)
        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
