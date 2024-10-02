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

        obs_space_low = np.zeros((vm_count, 4))
        obs_space_high = np.zeros((vm_count, 4))
        obs_space_high[0] = np.inf
        obs_space_high[1] = 1
        obs_space_high[2] = np.inf
        obs_space_high[3] = np.inf
        self.observation_space = spaces.Box(low=obs_space_low.flatten(), high=obs_space_high.flatten())
        self.action_space = spaces.Discrete(vm_count)

    # ----------------------- Reset method ----------------------------------------------------------------------------

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.task_action = self._calc_task_action(obs)
        new_obs = self._transform_observation(obs)
        return new_obs, info

    # ----------------------- Step method -----------------------------------------------------------------------------

    def step(self, action: Any) -> tuple[np.ndarray | None, float, bool, bool, dict[str, Any]]:  # type: ignore[override]
        action_dict = {"vm_id": action, "task_id": self.task_action}
        obs, reward, terminated, truncated, info = super().step(action_dict)
        if terminated or truncated:
            return None, reward, terminated, truncated, info

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
                obs["vm_completion_time"],
                obs["task_vm_compatibility"][self.task_action],
                obs["task_vm_time_cost"][self.task_action],
                obs["task_vm_power_cost"][self.task_action],
            ]
        ).T.flatten()
