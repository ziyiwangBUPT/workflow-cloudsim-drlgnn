from dataclasses import dataclass

import numpy as np
import torch


class GinEAgentMapper:
    def __init__(self, obs_size: int):
        self.obs_size = obs_size

    def map(
        self,
        task_state_scheduled: np.ndarray,
        task_state_ready: np.ndarray,
        task_completion_time: np.ndarray,
        vm_completion_time: np.ndarray,
        task_vm_compatibility: np.ndarray,
        task_vm_time_cost: np.ndarray,
        task_vm_energy_cost: np.ndarray,
        adj: np.ndarray,
    ) -> np.ndarray:
        num_tasks = task_vm_compatibility.shape[0]
        num_vms = task_vm_compatibility.shape[1]

        arr = np.concatenate(
            [
                np.array([num_tasks, num_vms], dtype=np.int32),  # Header
                np.array(task_state_scheduled, dtype=np.int32),  # num_tasks
                np.array(task_state_ready, dtype=np.int32),  # num_tasks
                np.array(task_completion_time, dtype=np.float64),  # num_tasks
                np.array(vm_completion_time, dtype=np.float64),  # num_vms
                np.array(task_vm_compatibility.flatten(), dtype=np.int32),  # num_tasks*num_vms
                np.array(task_vm_time_cost.flatten(), dtype=np.float64),  # num_tasks*num_vms
                np.array(task_vm_energy_cost.flatten(), dtype=np.float64),  # num_tasks*num_vms
                np.array(adj.flatten(), dtype=np.int32),  # num_tasks*num_tasks
            ]
        )

        assert len(arr) <= self.obs_size, "Observation size does not fit the buffer, please adjust the size of mapper"
        arr = np.pad(arr, (0, self.obs_size - len(arr)), "constant")

        return arr

    def unmap(self, tensor: torch.Tensor) -> "GinEAgentObsTensor":
        assert len(tensor) == self.obs_size, "Tensor size is not of expected size"

        num_tasks = int(tensor[0].long().item())
        num_vms = int(tensor[1].long().item())
        tensor = tensor[2:]

        task_state_scheduled = tensor[:num_tasks].long()
        tensor = tensor[num_tasks:]
        task_state_ready = tensor[:num_tasks].long()
        tensor = tensor[num_tasks:]
        task_completion_time = tensor[:num_tasks]
        tensor = tensor[num_tasks:]

        vm_completion_time = tensor[:num_vms]
        tensor = tensor[num_vms:]

        task_vm_compatibility = tensor[: num_tasks * num_vms].reshape(num_tasks, num_vms).long()
        tensor = tensor[num_tasks * num_vms :]
        task_vm_time_cost = tensor[: num_tasks * num_vms].reshape(num_tasks, num_vms)
        tensor = tensor[num_tasks * num_vms :]
        task_vm_energy_cost = tensor[: num_tasks * num_vms].reshape(num_tasks, num_vms)
        tensor = tensor[num_tasks * num_vms :]

        adj = tensor[: num_tasks * num_tasks].reshape(num_tasks, num_tasks).long()
        tensor = tensor[num_tasks * num_tasks :]

        assert not tensor.any(), "There are non-zero elements in the padding"

        return GinEAgentObsTensor(
            task_state_scheduled=task_state_scheduled,
            task_state_ready=task_state_ready,
            task_completion_time=task_completion_time,
            vm_completion_time=vm_completion_time,
            task_vm_compatibility=task_vm_compatibility,
            task_vm_time_cost=task_vm_time_cost,
            task_vm_energy_cost=task_vm_energy_cost,
            adj=adj,
        )


@dataclass
class GinEAgentObsTensor:
    task_state_scheduled: torch.Tensor
    task_state_ready: torch.Tensor
    task_completion_time: torch.Tensor
    vm_completion_time: torch.Tensor
    task_vm_compatibility: torch.Tensor
    task_vm_time_cost: torch.Tensor
    task_vm_energy_cost: torch.Tensor
    adj: torch.Tensor
