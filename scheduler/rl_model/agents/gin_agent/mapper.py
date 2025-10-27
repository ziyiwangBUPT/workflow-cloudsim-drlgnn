from dataclasses import dataclass

import numpy as np
import torch


class GinAgentMapper:
    def __init__(self, obs_size: int):
        self.obs_size = obs_size

    def map(
        self,
        task_state_scheduled: np.ndarray,
        task_state_ready: np.ndarray,
        task_length: np.ndarray,  # 保留：任务计算量
        task_normalized_deadline: np.ndarray,  # 替换 task_completion_time: Min-Max 归一化的子截止时间
        vm_speed: np.ndarray,
        vm_energy_rate: np.ndarray,
        vm_completion_time: np.ndarray,
        vm_carbon_intensity: np.ndarray,  # 新增：VM的碳强度特征
        task_dependencies: np.ndarray,
        compatibilities: np.ndarray,
    ) -> np.ndarray:
        num_tasks = task_length.shape[0]
        num_vms = vm_completion_time.shape[0]
        num_task_deps = task_dependencies.shape[1]
        num_compatibilities = compatibilities.shape[1]

        arr = np.concatenate(
            [
                np.array([num_tasks, num_vms, num_task_deps, num_compatibilities], dtype=np.int32),  # Header
                np.array(task_state_scheduled, dtype=np.int32),  # num_tasks
                np.array(task_state_ready, dtype=np.int32),  # num_tasks
                np.array(task_length, dtype=np.float64),  # num_tasks: 保留任务计算量
                np.array(task_normalized_deadline, dtype=np.float64),  # num_tasks: Min-Max 归一化的子截止时间
                np.array(vm_speed, dtype=np.float64),  # num_vms
                np.array(vm_energy_rate, dtype=np.float64),  # num_vms
                np.array(vm_completion_time, dtype=np.float64),  # num_vms
                np.array(vm_carbon_intensity, dtype=np.float64),  # num_vms: 新增碳强度特征
                np.array(task_dependencies.flatten(), dtype=np.int32),  # num_task_deps*2
                np.array(compatibilities.flatten(), dtype=np.int32),  # num_compatibilities*2
            ]
        )

        assert len(arr) <= self.obs_size, "Observation size does not fit the buffer, please adjust the size of mapper"
        arr = np.pad(arr, (0, self.obs_size - len(arr)), "constant")

        return arr

    def unmap(self, tensor: torch.Tensor) -> "GinAgentObsTensor":
        assert len(tensor) == self.obs_size, "Tensor size is not of expected size"

        num_tasks = int(tensor[0].long().item())
        num_vms = int(tensor[1].long().item())
        num_task_deps = int(tensor[2].long().item())
        num_compatibilities = int(tensor[3].long().item())
        tensor = tensor[4:]

        task_state_scheduled = tensor[:num_tasks].long()
        tensor = tensor[num_tasks:]
        task_state_ready = tensor[:num_tasks].long()
        tensor = tensor[num_tasks:]
        task_length = tensor[:num_tasks]  # 保留：任务计算量
        tensor = tensor[num_tasks:]
        task_normalized_deadline = tensor[:num_tasks]  # Min-Max 归一化的子截止时间
        tensor = tensor[num_tasks:]

        vm_speed = tensor[:num_vms]
        tensor = tensor[num_vms:]
        vm_energy_rate = tensor[:num_vms]
        tensor = tensor[num_vms:]
        vm_completion_time = tensor[:num_vms]
        tensor = tensor[num_vms:]
        vm_carbon_intensity = tensor[:num_vms]  # 新增：碳强度特征
        tensor = tensor[num_vms:]

        task_dependencies = tensor[: num_task_deps * 2].reshape(2, num_task_deps).long()
        tensor = tensor[num_task_deps * 2 :]
        compatibilities = tensor[: num_compatibilities * 2].reshape(2, num_compatibilities).long()
        tensor = tensor[num_compatibilities * 2 :]

        assert not tensor.any(), "There are non-zero elements in the padding"

        return GinAgentObsTensor(
            task_state_scheduled=task_state_scheduled,
            task_state_ready=task_state_ready,
            task_length=task_length,  # 保留
            task_normalized_deadline=task_normalized_deadline,  # 替换 task_completion_time
            vm_speed=vm_speed,
            vm_energy_rate=vm_energy_rate,
            vm_completion_time=vm_completion_time,
            vm_carbon_intensity=vm_carbon_intensity,  # 新增
            task_dependencies=task_dependencies,
            compatibilities=compatibilities,
        )


@dataclass
class GinAgentObsTensor:
    task_state_scheduled: torch.Tensor
    task_state_ready: torch.Tensor
    task_length: torch.Tensor  # 保留：任务计算量
    task_normalized_deadline: torch.Tensor  # Min-Max 归一化的子截止时间（替换 task_completion_time）
    vm_speed: torch.Tensor
    vm_energy_rate: torch.Tensor
    vm_completion_time: torch.Tensor
    vm_carbon_intensity: torch.Tensor  # 新增：VM的碳强度特征
    task_dependencies: torch.Tensor
    compatibilities: torch.Tensor
