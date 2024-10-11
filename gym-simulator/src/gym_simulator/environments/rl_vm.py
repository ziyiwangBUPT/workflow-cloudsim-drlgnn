import copy

from typing import Any
from gymnasium import spaces
import numpy as np

from gym_simulator.environments.rl import RlCloudSimEnvironment


class RlVmCloudSimEnvironment(RlCloudSimEnvironment):
    """
    A RL environment for the CloudSim simulator with VM selection as the action space.
    The task action is the task with the minimum completion time among the ready tasks.
    """

    task_action: int = 0

    def __init__(self, env_config: dict[str, Any]):
        # Override args
        super().__init__(env_config)
        self.parent_observation_space = copy.deepcopy(self.observation_space)
        self.parent_action_space = copy.deepcopy(self.action_space)

        vm_count = self.vm_count

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4 * vm_count,), dtype=np.float32)
        self.action_space = spaces.Discrete(vm_count)

    # ----------------------- Reset method ----------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.task_action = self._calc_task_action(obs)
        new_obs = self._transform_observation(obs)
        return new_obs, info

    # ----------------------- Step method -----------------------------------------------------------------------------

    def step(self, action: Any):
        action_dict = {"vm_id": action, "task_id": self.task_action}
        obs, reward, terminated, truncated, info = super().step(action_dict)
        if terminated or truncated:
            return self.observation_space.sample(), reward, terminated, truncated, info

        self.task_action = self._calc_task_action(obs)
        new_obs = self._transform_observation(obs)
        return new_obs, reward, terminated, truncated, info

    def _calc_task_action(self, obs: dict[str, Any]) -> int:
        task_state_ready = obs["task_state_ready"]
        task_completion_time = obs["task_completion_time"]

        # Task ID is the ready task with the minimum completion time
        ready_task_ids = np.where(task_state_ready == 1)
        min_comp_time_of_ready_tasks = np.min(task_completion_time[ready_task_ids])
        return np.where(task_completion_time == min_comp_time_of_ready_tasks)[0][0]

    def _transform_observation(self, obs: dict[str, Any]) -> np.ndarray:
        return np.vstack(
            [
                obs["task_vm_compatibility"][self.task_action],
                obs["vm_completion_time"],
                obs["task_vm_time_cost"][self.task_action],
                obs["task_vm_power_cost"][self.task_action],
            ]
        ).flatten()
