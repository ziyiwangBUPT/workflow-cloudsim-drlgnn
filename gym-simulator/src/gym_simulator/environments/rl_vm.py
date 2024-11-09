import copy

from typing import Any
from gymnasium import spaces
import numpy as np

from icecream import ic

from gym_simulator.environments.rl import RlCloudSimEnvironment


class RlVmCloudSimEnvironment(RlCloudSimEnvironment):
    """
    A RL environment for the CloudSim simulator with VM selection as the action space.
    The task action is the task with the minimum completion time among the ready tasks.
    """

    def __init__(self, env_config: dict[str, Any]):
        # Override args
        super().__init__(env_config)
        self.parent_observation_space = copy.deepcopy(self.observation_space)
        self.parent_action_space = copy.deepcopy(self.action_space)

        max_tasks = (self.task_limit + 2) * self.workflow_count
        max_machines = self.vm_count
        self.observation_space_size = (
            2  # headers
            + max_tasks  # task_state_scheduled
            + max_tasks  # task_state_ready
            + max_tasks  # task_completion_time
            + max_machines  # vm_completion_time
            + max_machines * max_tasks  # task_vm_compatibility
            + max_machines * max_tasks  # task_vm_time_cost
            + max_machines * max_tasks  # task_vm_power_cost
            + max_tasks**2  # adjacency matrix
        )
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.observation_space_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.vm_count)

    # ----------------------- Reset method ----------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        new_obs = self._transform_observation(obs)
        self.prev_obs = obs
        return new_obs, info

    # ----------------------- Step method -----------------------------------------------------------------------------

    def step(self, action: Any):
        # VM with the minimum completion time amoung compatible VMs
        compatible_vms = self.prev_obs["task_vm_compatibility"][action]
        vm_completion_times = self.prev_obs["vm_completion_time"]
        vm_completion_times = np.where(compatible_vms, vm_completion_times, float("inf"))
        vm_action = np.argmin(vm_completion_times)

        action_dict = {"vm_id": vm_action, "task_id": action}
        ic(action_dict)

        obs, reward, terminated, truncated, info = super().step(action_dict)
        if terminated or truncated:
            error = info.get("error")
            ic("Terminated", error)
            return np.zeros(self.observation_space_size), reward, terminated, truncated, info

        new_obs = self._transform_observation(obs)
        self.prev_obs = obs
        return new_obs, reward, terminated, truncated, info

    def _transform_observation(self, obs: dict[str, Any]) -> np.ndarray:
        num_tasks = obs["task_completion_time"].size
        num_machines = self.vm_count

        arr = np.concatenate(
            [
                [num_tasks, num_machines],
                np.array(obs["task_state_scheduled"], dtype=np.int32).flatten(),
                np.array(obs["task_state_ready"], dtype=np.int32).flatten(),
                np.array(obs["task_completion_time"]).flatten(),
                np.array(obs["vm_completion_time"]).flatten(),
                np.array(obs["task_vm_compatibility"]).flatten(),
                np.array(obs["task_vm_time_cost"]).flatten(),
                np.array(obs["task_vm_power_cost"]).flatten(),
                np.array(obs["task_graph_edges"], dtype=np.int32).flatten(),
            ]
        )

        ic(num_tasks, num_machines)
        if self.observation_space_size > len(arr):
            return np.pad(arr, (0, self.observation_space_size - len(arr)), "constant")

        return arr
