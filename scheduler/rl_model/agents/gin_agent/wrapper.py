from typing import SupportsFloat, Any

import numpy as np
import gymnasium as gym

from scheduler.config.settings import MAX_OBS_SIZE
from scheduler.rl_model.agents.gin_agent.mapper import GinAgentMapper
from scheduler.rl_model.core.env.action import EnvAction
from scheduler.rl_model.core.env.observation import EnvObservation
from scheduler.rl_model.core.utils.helpers import active_energy_consumption_per_mi
from scheduler.config.carbon_intensity import get_future_6h_carbon_intensity_curve, FIXED_NUM_HOSTS


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

        # 奖励函数：参考 workflow-cloudsim-drlgnn-master 的相对改善量模式
        # reward = makespan_reward + carbon_reward
        # 其中 makespan_reward = -(Δmakespan / makespan)
        #      carbon_reward = -(Δcarbon / carbon)

        makespan_reward = -(obs.makespan() - self.prev_obs.makespan()) / obs.makespan()

        # 使用 carbon_cost_optimistic() 替代 energy_consumption()
        # 计算碳成本的乐观估计（已调度任务用实际值，未调度任务用最低碳强度估计）
        current_carbon = obs.carbon_cost_optimistic()
        prev_carbon = self.prev_obs.carbon_cost_optimistic()

        if current_carbon > 0:
            carbon_reward = -(current_carbon - prev_carbon) / current_carbon
        else:
            carbon_reward = 0.0

        reward = makespan_reward + carbon_reward

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

        # VM未来6小时碳强度曲线特征
        # 从VM完成时间开始，获取未来6小时的碳强度曲线（替换原来的单个碳强度值）
        vm_carbon_intensity_curve_6h = np.array([
            get_future_6h_carbon_intensity_curve(
                host_id=vm.host_id % FIXED_NUM_HOSTS,
                start_time=vm.completion_time
            )
            for vm in observation.vm_observations
        ])  # shape: (num_vms, 6)

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
            vm_carbon_intensity_curve_6h=vm_carbon_intensity_curve_6h,  # 未来6小时碳强度曲线（替换单个碳强度值）
            task_dependencies=task_dependencies,
            compatibilities=compatibilities,
        )
