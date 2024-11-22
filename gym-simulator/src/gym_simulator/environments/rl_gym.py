import copy

from typing import Any
from gymnasium import spaces
import numpy as np

from icecream import ic

from gym_simulator.environments.rl import RlCloudSimEnvironment


class RlGymCloudSimEnvironment(RlCloudSimEnvironment):
    """
    A RL environment for the CloudSim simulator with VM selection as the action space.
    The task action is the task with the minimum completion time among the ready tasks.
    """

    max_obs_size = 100000

    def __init__(self, env_config: dict[str, Any]):
        # Override args
        super().__init__(env_config)
        self.parent_observation_space = copy.deepcopy(self.observation_space)
        self.parent_action_space = copy.deepcopy(self.action_space)

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.max_obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(42 * 42)

    # ----------------------- Reset method ----------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        new_obs = self._transform_observation(obs)
        self.prev_obs = obs
        return new_obs, info

    # ----------------------- Step method -----------------------------------------------------------------------------

    def step(self, action: Any):
        # VM with the minimum completion time amoung compatible VMs
        num_vms = self.prev_obs["task_vm_compatibility"].shape[1]
        action_dict = {"vm_id": action % num_vms, "task_id": action // num_vms}

        obs, reward, terminated, truncated, info = super().step(action_dict)
        if terminated or truncated:
            error = info.get("error")
            if error is not None:
                ic("Terminated", error)
            return np.zeros(42), reward, terminated, truncated, info

        new_obs = self._transform_observation(obs)
        self.prev_obs = obs
        return new_obs, reward, terminated, truncated, info

    def _transform_observation(self, obs: dict[str, Any]) -> np.ndarray:
        num_tasks = obs["task_vm_compatibility"].shape[0]
        num_vms = obs["task_vm_compatibility"].shape[1]

        arr = np.concatenate(
            [
                [num_tasks, num_vms],
                np.array(obs["task_state_scheduled"], dtype=np.int32),
                np.array(obs["task_state_ready"], dtype=np.int32),
                np.array(obs["task_completion_time"]),
                np.array(obs["vm_completion_time"]),
                np.array(obs["task_vm_compatibility"]).flatten(),
                np.array(obs["task_vm_time_cost"]).flatten(),
                np.array(obs["task_vm_power_cost"]).flatten(),
                np.array(obs["task_graph_edges"]).flatten(),
            ]
        )

        assert len(arr) <= self.max_obs_size, "Observation size does not fit the buffer, please adjust the max_obs_size"
        return np.pad(arr, (0, self.max_obs_size - len(arr)), "constant")
