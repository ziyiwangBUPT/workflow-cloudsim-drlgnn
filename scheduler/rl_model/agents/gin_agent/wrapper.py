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
    prev_makespan: float

    def __init__(self, env: gym.Env[np.ndarray, int]):
        super().__init__(env)
        self.mapper = GinAgentMapper(MAX_OBS_SIZE)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        assert isinstance(obs, EnvObservation)
        self.vm_count = len(obs.vm_observations)
        makespan, mapped_obs = self.map_observation(obs)
        self.prev_makespan = makespan
        return mapped_obs, info

    def step(self, action: int) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        mapped_action = self.map_action(action)
        obs, reward, terminated, truncated, info = super().step(mapped_action)

        assert isinstance(obs, EnvObservation)
        self.vm_count = len(obs.vm_observations)
        makespan, mapped_obs = self.map_observation(obs)
        reward = -(makespan - self.prev_makespan) / makespan
        self.prev_makespan = makespan
        return mapped_obs, reward, terminated, truncated, info

    def map_action(self, action: int) -> EnvAction:
        return EnvAction(task_id=int(action // self.vm_count), vm_id=int(action % self.vm_count))

    def map_observation(self, observation: EnvObservation) -> tuple[float, np.ndarray]:
        # Task observations
        task_assignments = np.array([task.assigned_vm_id or 0 for task in observation.task_observations])
        task_state_scheduled = np.array([task.assigned_vm_id is not None for task in observation.task_observations])
        task_state_ready = np.array([task.is_ready for task in observation.task_observations])
        task_lengths = np.array([task.length for task in observation.task_observations])

        # VM observations
        vm_speeds = np.array([vm.cpu_speed_mips for vm in observation.vm_observations])
        vm_energy_rates = np.array([energy_consumption_per_mi(vm) for vm in observation.vm_observations])
        vm_completion_times = np.array([vm.completion_time for vm in observation.vm_observations])

        # Task-Task observations
        task_dependencies = np.array(observation.task_dependencies).T

        # Task-VM observations
        compatibilities = np.array(observation.compatibilities).T

        # Calculate completion times iteratively (task dependencies)
        task_completion_time = [task.completion_time for task in observation.task_observations]
        for task_id in range(len(observation.task_observations)):
            child_ids = [cid for pid, cid in observation.task_dependencies if pid == task_id]
            for child_id in child_ids:
                if observation.task_observations[child_id].assigned_vm_id is not None:
                    continue
                # Since following are not scheduled yet, only dependencies will be from parent
                assert task_id < child_id, "DAG property violation"
                child_available_vm_speeds = vm_speeds[compatibilities[1][compatibilities[0] == child_id]]
                child_execution_cost = task_lengths[child_id] / child_available_vm_speeds.max()
                task_completion_time[child_id] = max(
                    task_completion_time[child_id], task_completion_time[task_id] + child_execution_cost
                )

        mapped_obs = self.mapper.map(
            task_assignments=task_assignments,
            task_state_scheduled=task_state_scheduled,
            task_state_ready=task_state_ready,
            task_lengths=task_lengths,
            vm_speeds=vm_speeds,
            vm_energy_rates=vm_energy_rates,
            vm_completion_times=vm_completion_times,
            task_dependencies=task_dependencies,
            compatibilities=compatibilities,
        )

        return task_completion_time[-1], mapped_obs
