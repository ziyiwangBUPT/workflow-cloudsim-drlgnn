import copy
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
from gym_simulator.algorithms.base import BaseScheduler
from gym_simulator.algorithms.rl_agents.gin_agent import GinAgent
from gym_simulator.algorithms.rl_agents.input_decoder import DecodedObservation, decode_observation
from gym_simulator.algorithms.rl_agents.mpgn_agent import MpgnAgent
from gym_simulator.core.simulators.proxy import InternalProxySimulatorObs
from gym_simulator.core.types import TaskDto, VmAssignmentDto, VmDto
from gym_simulator.environments.rl_gym import RlGymCloudSimEnvironment


class RlHeuristicScheduler(BaseScheduler):
    def __init__(self, env_config: dict[str, Any], heuristic: str):
        self.env_config = env_config
        self.heuristic = heuristic

    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        # Load agent and obs state
        self.env_config["simulator_kwargs"]["proxy_obs"].tasks = tasks
        self.env_config["simulator_kwargs"]["proxy_obs"].vms = vms

        env = RlGymCloudSimEnvironment(env_config=copy.deepcopy(self.env_config))
        next_obs, _ = env.reset(seed=self.env_config["seed"])
        while True:
            obs_tensor = torch.Tensor(next_obs.flatten()).to("cpu")
            obs = decode_observation(obs_tensor)
            (task_id, vm_id) = self.get_action(obs)
            action = task_id * obs.vm_completion_time.shape[0] + vm_id
            next_obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        self.vm_completion_time = env.state.vm_completion_time.copy()
        assert len(tasks) == len(info["vm_assignments"])
        return info["vm_assignments"]

    def get_action(self, obs: DecodedObservation) -> Tuple[int, int]:
        if self.heuristic == "makespan":
            # Find ready task index with minimum task completion time
            # Find the VM that gives minimum completion time for that task
            ready_task_indices = np.where(obs.task_state_ready)[0]
            task_completion_times = obs.task_completion_time[ready_task_indices]
            min_task_index = ready_task_indices[np.argmin(task_completion_times)]

            compatible_vm_indices = np.where(obs.task_vm_compatibility[min_task_index])[0]
            vm_completion_times = obs.task_vm_time_cost[min_task_index][compatible_vm_indices]
            min_vm_index = compatible_vm_indices[np.argmin(vm_completion_times)]

            assert obs.task_vm_compatibility[min_task_index][min_vm_index] == 1
            return min_task_index, min_vm_index

        elif self.heuristic == "energy":
            # Find the ready task index with minimum energy cost
            ready_task_indices = np.where(obs.task_state_ready)[0]
            min_energy_cost = float("inf")
            min_task_index = -1
            min_vm_index = -1

            for task_index in ready_task_indices:
                compatible_vm_indices = np.where(obs.task_vm_compatibility[task_index])[0]

                # Calculate energy cost for compatible VMs
                vm_energy_costs = obs.task_vm_power_cost[task_index, compatible_vm_indices]

                # Find the minimum energy cost for this task
                task_min_vm_index = compatible_vm_indices[np.argmin(vm_energy_costs)]
                task_min_energy_cost = vm_energy_costs[np.argmin(vm_energy_costs)]
                if task_min_energy_cost < min_energy_cost:
                    min_energy_cost = task_min_energy_cost
                    min_task_index = task_index
                    min_vm_index = task_min_vm_index

            assert obs.task_vm_compatibility[min_task_index][min_vm_index] == 1
            return min_task_index, min_vm_index
        raise NotImplementedError()
