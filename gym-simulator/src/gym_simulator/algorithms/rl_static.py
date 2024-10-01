import copy
from typing import Any

import numpy as np
from gym_simulator.algorithms.base import BaseScheduler
from gym_simulator.core.types import TaskDto, VmAssignmentDto, VmDto
from gym_simulator.environments.rl import RlCloudSimEnvironment


class RlStaticScheduler(BaseScheduler):
    """
    RL Env based scheduler.

    This scheduler runs the RL environment instance internally to schedule the tasks.
    The algorithm is not a reinforcement learning algorithm, just a simple heuristic algorithm.
    """

    def __init__(self, env_config: dict[str, Any]):
        self.env_config = env_config

    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        env = RlCloudSimEnvironment(env_config=copy.deepcopy(self.env_config))

        obs, _ = env.reset()
        while True:
            task_state_ready = obs["task_state_ready"]
            task_completion_time = obs["task_completion_time"]
            vm_completion_time = obs["vm_completion_time"]
            task_vm_compatibility = obs["task_vm_compatibility"]
            task_vm_time_cost = obs["task_vm_time_cost"]
            task_graph_edges = obs["task_graph_edges"]

            assert len(tasks) + 2 == task_vm_compatibility.shape[0]
            assert len(vms) == task_vm_compatibility.shape[1]

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
                possible_start_time = max(max_parent_comp_time, vm_completion_time[vm_id])
                comp_time = possible_start_time + task_vm_time_cost[next_task_id, vm_id]
                if comp_time < min_comp_time:
                    min_comp_time = comp_time
                    next_vm_id = vm_id
            assert min_comp_time == min_comp_time_of_ready_tasks

            action = {"vm_id": next_vm_id, "task_id": next_task_id}
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        return info["vm_assignments"]
