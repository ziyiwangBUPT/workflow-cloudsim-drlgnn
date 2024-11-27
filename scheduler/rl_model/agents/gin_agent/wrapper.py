from typing import SupportsFloat, Any

import numpy as np
import gymnasium as gym
from icecream import ic

from scheduler.config.settings import MAX_OBS_SIZE
from scheduler.rl_model.agents.gin_agent.mapper import GinAgentMapper
from scheduler.rl_model.core.env.action import EnvAction
from scheduler.rl_model.core.env.observation import EnvObservation
from scheduler.rl_model.core.utils.helpers import active_energy_consumption_per_mi


class GinAgentWrapper(gym.Wrapper):
    observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(MAX_OBS_SIZE,), dtype=np.float32)
    action_space = gym.spaces.Discrete(MAX_OBS_SIZE)

    prev_obs: EnvObservation

    def __init__(self, env: gym.Env[np.ndarray, int]):
        super().__init__(env)
        self.mapper = GinAgentMapper(MAX_OBS_SIZE)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        assert isinstance(obs, EnvObservation)
        mapped_obs = self.map_observation(obs)

        self.prev_obs = obs
        return mapped_obs, info

    def step(self, action: int) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        mapped_action = self.map_action(action)
        obs, _, terminated, truncated, info = super().step(mapped_action)
        assert isinstance(obs, EnvObservation)
        mapped_obs = self.map_observation(obs)

        makespan_reward = -(obs.makespan() - self.prev_obs.makespan()) / obs.makespan()
        energy_reward = -(obs.energy_consumption() - self.prev_obs.energy_consumption()) / obs.energy_consumption()
        reward = makespan_reward + energy_reward

        self.prev_obs = obs
        return mapped_obs, reward, terminated, truncated, info

    def map_action(self, action: int) -> EnvAction:
        vm_count = len(self.prev_obs.vm_observations)
        return EnvAction(task_id=int(action // vm_count), vm_id=int(action % vm_count))

    def map_observation(self, observation: EnvObservation) -> np.ndarray:
        # Task observations
        task_state_scheduled = np.array([task.assigned_vm_id is not None for task in observation.task_observations])
        task_state_ready = np.array([task.is_ready for task in observation.task_observations])
        task_length = np.array([task.length for task in observation.task_observations])

        # VM observations
        vm_speed = np.array([vm.cpu_speed_mips for vm in observation.vm_observations])
        vm_energy_rate = np.array([active_energy_consumption_per_mi(vm) for vm in observation.vm_observations])
        vm_completion_time = np.array([vm.completion_time for vm in observation.vm_observations])

        # Task-Task observations
        task_dependencies = np.array(observation.task_dependencies).T

        # Task-VM observations
        compatibilities = np.array(observation.compatibilities).T

        # Task completion times
        task_completion_time = observation.task_completion_time()
        assert task_completion_time is not None

        return self.mapper.map(
            task_state_scheduled=task_state_scheduled,
            task_state_ready=task_state_ready,
            task_length=task_length,
            task_completion_time=task_completion_time,
            vm_speed=vm_speed,
            vm_energy_rate=vm_energy_rate,
            vm_completion_time=vm_completion_time,
            task_dependencies=task_dependencies,
            compatibilities=compatibilities,
        )
