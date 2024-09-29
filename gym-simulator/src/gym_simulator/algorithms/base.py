from abc import ABC, abstractmethod
from gym_simulator.algorithms.types import VmDto, TaskDto, VmAssignmentDto


class BaseScheduler(ABC):
    """Base class for scheduling algorithms."""

    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        """Schedule the tasks on the VMs."""
        raise NotImplementedError

    def is_vm_suitable(self, vm: VmDto, task: TaskDto) -> bool:
        """Check if the VM is suitable for the task."""
        return vm.memory_mb >= task.req_memory_mb

    def is_optimal(self) -> bool:
        """Check if the scheduling algorithm is optimal."""
        return False
