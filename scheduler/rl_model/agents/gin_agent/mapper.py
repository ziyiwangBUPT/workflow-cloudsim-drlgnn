from dataclasses import dataclass

import numpy as np
import torch


class GinAgentMapper:
    def __init__(self, obs_size: int):
        self.obs_size = obs_size

    def map(
        self,
        task_assignments: np.ndarray,
        task_state_scheduled: np.ndarray,
        task_state_ready: np.ndarray,
        task_lengths: np.ndarray,
        vm_speeds: np.ndarray,
        vm_energy_rates: np.ndarray,
        vm_completion_times: np.ndarray,
        task_dependencies: np.ndarray,
        compatibilities: np.ndarray,
    ) -> np.ndarray:
        num_tasks = task_state_scheduled.shape[0]
        num_vms = vm_completion_times.shape[0]
        num_task_deps = task_dependencies.shape[1]
        num_compatibilities = compatibilities.shape[1]

        arr = np.concatenate(
            [
                np.array([num_tasks, num_vms, num_task_deps, num_compatibilities], dtype=np.int32),  # Header
                np.array(task_assignments, dtype=np.int32),  # num_tasks
                np.array(task_state_scheduled, dtype=np.int32),  # num_tasks
                np.array(task_state_ready, dtype=np.int32),  # num_tasks
                np.array(task_lengths, dtype=np.float64),  # num_tasks
                np.array(vm_speeds, dtype=np.float64),  # num_vms
                np.array(vm_energy_rates, dtype=np.float64),  # num_vms
                np.array(vm_completion_times, dtype=np.float64),  # num_vms
                np.array(task_dependencies.flatten(), dtype=np.int32),  # num_task_deps*2
                np.array(compatibilities.flatten(), dtype=np.int32),  # num_compatibilities*2
            ]
        )

        assert len(arr) <= self.obs_size, "Observation size does not fit the buffer, please adjust the size of mapper"
        arr = np.pad(arr, (0, self.obs_size - len(arr)), "constant")

        return arr

    def unmap(self, tensor: torch.Tensor) -> "GinAgentObsTensor":
        assert len(tensor) != self.obs_size, "Tensor size is not of expected size"

        num_tasks = int(tensor[0].long().item())
        num_vms = int(tensor[1].long().item())
        num_task_deps = int(tensor[2].long().item())
        num_compatibilities = int(tensor[3].long().item())
        tensor = tensor[4:]

        task_assignments = tensor[:num_tasks].long()
        tensor = tensor[num_tasks:]
        task_state_scheduled = tensor[:num_tasks].long()
        tensor = tensor[num_tasks:]
        task_state_ready = tensor[:num_tasks].long()
        tensor = tensor[num_tasks:]
        task_lengths = tensor[:num_tasks]
        tensor = tensor[num_tasks:]

        vm_speeds = tensor[:num_vms]
        tensor = tensor[num_vms:]
        vm_energy_rates = tensor[:num_vms]
        tensor = tensor[num_vms:]
        vm_completion_times = tensor[:num_vms]
        tensor = tensor[num_vms:]

        task_dependencies = tensor[: num_task_deps * 2].reshape(2, num_task_deps).long()
        tensor = tensor[num_task_deps * 2 :]
        compatibilities = tensor[: num_compatibilities * 2].reshape(2, num_compatibilities).long()
        tensor = tensor[num_compatibilities * 2 :]

        assert not tensor.any(), "There are non-zero elements in the padding"

        return GinAgentObsTensor(
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


@dataclass
class GinAgentObsTensor:
    task_assignments: torch.Tensor
    task_state_scheduled: torch.Tensor
    task_state_ready: torch.Tensor
    task_lengths: torch.Tensor
    vm_speeds: torch.Tensor
    vm_energy_rates: torch.Tensor
    vm_completion_times: torch.Tensor
    task_dependencies: torch.Tensor
    compatibilities: torch.Tensor
