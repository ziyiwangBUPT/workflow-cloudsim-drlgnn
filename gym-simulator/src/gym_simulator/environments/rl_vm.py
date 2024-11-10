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

        self.max_tasks = (self.task_limit + 2) * self.workflow_count
        self.observation_space_size = (
            2  # headers
            + self.max_tasks  # task_state_scheduled
            + self.max_tasks  # task_state_ready
            + self.max_tasks  # task_completion_time
            + self.vm_count  # vm_completion_time
            + self.vm_count * self.max_tasks  # task_vm_compatibility
            + self.vm_count * self.max_tasks  # task_vm_time_cost
            + self.vm_count * self.max_tasks  # task_vm_power_cost
            + self.max_tasks**2  # adjacency matrix
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
        action_dict = {"vm_id": action % self.vm_count, "task_id": action // self.vm_count}

        obs, reward, terminated, truncated, info = super().step(action_dict)
        if terminated or truncated:
            error = info.get("error")
            if error is not None:
                ic("Terminated", error)
            return np.zeros(self.observation_space_size), reward, terminated, truncated, info

        new_obs = self._transform_observation(obs)
        self.prev_obs = obs
        return new_obs, reward, terminated, truncated, info

    def _transform_observation(self, obs: dict[str, Any]) -> np.ndarray:
        num_tasks = obs["task_state_scheduled"].shape[0]

        task_pad = (0, self.max_tasks - num_tasks)
        machine_pad = (0, 0)  # Since max_machines = vm_count

        arr = np.concatenate(
            [
                [self.max_tasks, self.vm_count],
                np.pad(np.array(obs["task_state_scheduled"], dtype=np.int32), task_pad),
                np.pad(np.array(obs["task_state_ready"], dtype=np.int32), task_pad),
                np.pad(np.array(obs["task_completion_time"]), task_pad),
                np.pad(np.array(obs["vm_completion_time"]), machine_pad),
                np.pad(np.array(obs["task_vm_compatibility"]), (task_pad, machine_pad)).flatten(),
                np.pad(np.array(obs["task_vm_time_cost"]), (task_pad, machine_pad)).flatten(),
                np.pad(np.array(obs["task_vm_power_cost"]), (task_pad, machine_pad)).flatten(),
                np.pad(np.array(obs["task_graph_edges"]), (task_pad, task_pad)).flatten(),
            ]
        )

        if self.observation_space_size > len(arr):
            return np.pad(arr, (0, self.observation_space_size - len(arr)), "constant")

        return arr
