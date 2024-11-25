from typing import SupportsFloat, Any

import numpy as np
import gymnasium as gym

from scheduler.config.settings import MAX_OBS_SIZE
from scheduler.rl_model.agents.gin_agent.mapper import GinAgentMapper
from scheduler.rl_model.core.env.action import EnvAction
from scheduler.rl_model.core.env.observation import EnvObservation
from scheduler.rl_model.core.utils.helpers import energy_consumption_per_mi


class GinAgentWrapper(gym.Wrapper):
    observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(MAX_OBS_SIZE,), dtype=np.float32)
    action_space = gym.spaces.Discrete(MAX_OBS_SIZE)

    vm_count: int

    def __init__(self, env: gym.Env[np.ndarray, int]):
        super().__init__(env)
        self.mapper = GinAgentMapper(MAX_OBS_SIZE)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        mapped_obs = self.map_observation(obs)
        self.vm_count = int(mapped_obs[1])
        return mapped_obs, info

    def step(self, action: int) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        mapped_action = self.map_action(action)
        obs, reward, terminated, truncated, info = super().step(mapped_action)
        mapped_obs = self.map_observation(obs)
        self.vm_count = int(mapped_obs[1])
        return mapped_obs, reward, terminated, truncated, info

    def map_action(self, action: int) -> EnvAction:
        return EnvAction(task_id=int(action // self.vm_count), vm_id=int(action % self.vm_count))

    def map_observation(self, observation: EnvObservation) -> np.ndarray:
        # Task observations
        task_assignments = [task.assigned_vm_id or 0 for task in observation.task_observations]
        task_state_scheduled = [task.assigned_vm_id is not None for task in observation.task_observations]
        task_state_ready = [task.is_ready for task in observation.task_observations]
        task_lengths = [task.length for task in observation.task_observations]

        # VM observations
        vm_speeds = [vm.cpu_speed_mips for vm in observation.vm_observations]
        vm_energy_rates = [energy_consumption_per_mi(vm) for vm in observation.vm_observations]
        vm_completion_times = [vm.completion_time for vm in observation.vm_observations]

        # Task-Task observations
        task_dependencies = list(observation.task_dependencies)

        # Task-VM observations
        compatibilities = list(observation.compatibilities)

        return self.mapper.map(
            task_assignments=np.array(task_assignments),
            task_state_scheduled=np.array(task_state_scheduled),
            task_state_ready=np.array(task_state_ready),
            task_lengths=np.array(task_lengths),
            vm_speeds=np.array(vm_speeds),
            vm_energy_rates=np.array(vm_energy_rates),
            vm_completion_times=np.array(vm_completion_times),
            task_dependencies=np.array(task_dependencies).T,
            compatibilities=np.array(compatibilities).T,
        )
