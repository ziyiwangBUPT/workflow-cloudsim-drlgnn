from abc import ABC, abstractmethod
from gym_simulator.algorithms.types import VmDto, TaskDto, VmAssignmentDto


class BaseScheduler(ABC):
    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        raise NotImplementedError

    def is_vm_suitable(self, vm: VmDto, task: TaskDto) -> bool:
        return vm.memory_mb >= task.req_memory_mb
