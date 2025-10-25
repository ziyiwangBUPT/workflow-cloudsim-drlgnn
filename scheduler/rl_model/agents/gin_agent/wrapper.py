from typing import SupportsFloat, Any

import numpy as np
import gymnasium as gym

from scheduler.config.settings import MAX_OBS_SIZE
from scheduler.rl_model.agents.gin_agent.mapper import GinAgentMapper
from scheduler.rl_model.core.env.action import EnvAction
from scheduler.rl_model.core.env.observation import EnvObservation
from scheduler.rl_model.core.utils.helpers import active_energy_consumption_per_mi


class GinAgentWrapper(gym.Wrapper):
    observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(MAX_OBS_SIZE,), dtype=np.float32)
    action_space = gym.spaces.Discrete(MAX_OBS_SIZE)

    prev_obs: EnvObservation
    initial_obs: EnvObservation

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
        self.initial_obs = obs
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
        
        # 计算 Min-Max 归一化的子截止时间（替换 task_completion_time）
        # 参考 ecmws-experiments/tasks/workflow.py 的 make_stored_graph() 方法
        # 使用 Min-Max 归一化: (deadline - min) / (max - min)
        
        # 第1步：从当前 State 的所有任务中提取 deadline 值
        task_deadlines = np.array([task.deadline for task in observation.task_observations])
        
        # 第2步：动态计算 min 和 max
        min_deadline = task_deadlines.min()
        max_deadline = task_deadlines.max()
        delta_deadline = max_deadline - min_deadline
        
        # 第3步：Min-Max 归一化，处理除零异常
        eps = 1e-2  # 防止数值不稳定的小阈值（与 ecmws-experiments 一致）
        if delta_deadline <= eps:
            # 当所有任务的 deadline 相同或非常接近时，设为1
            task_normalized_deadline = np.ones_like(task_deadlines)
        else:
            # 标准 Min-Max 归一化: (x - min) / (max - min)
            task_normalized_deadline = (task_deadlines - min_deadline) / delta_deadline

        # VM observations
        vm_speed = np.array([vm.cpu_speed_mips for vm in observation.vm_observations])
        vm_energy_rate = np.array([active_energy_consumption_per_mi(vm) for vm in observation.vm_observations])
        vm_completion_time = np.array([vm.completion_time for vm in observation.vm_observations])

        # Task-Task observations
        task_dependencies = np.array(observation.task_dependencies).T

        # Task-VM observations
        compatibilities = np.array(observation.compatibilities).T

        return self.mapper.map(
            task_state_scheduled=task_state_scheduled,
            task_state_ready=task_state_ready,
            task_length=task_length,  # 保留：任务计算量
            task_normalized_deadline=task_normalized_deadline,  # 替换 task_completion_time
            vm_speed=vm_speed,
            vm_energy_rate=vm_energy_rate,
            vm_completion_time=vm_completion_time,
            task_dependencies=task_dependencies,
            compatibilities=compatibilities,
        )
