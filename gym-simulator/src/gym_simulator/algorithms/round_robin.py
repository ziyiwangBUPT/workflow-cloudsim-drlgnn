from typing import override

from gym_simulator.algorithms.base_ready_queue import BaseReadyQueueScheduler
from gym_simulator.algorithms.types import TaskDto, VmDto


class RoundRobinScheduler(BaseReadyQueueScheduler):
    """
    Implementation of the Round Robin scheduling algorithm.

    Round Robin is a simple scheduling algorithm that schedules the tasks in a circular order.
    """

    vm_index: int = 0

    @override
    def select_task(self, ready_tasks: list[TaskDto]) -> TaskDto:
        """Choose the next task (with no preference)."""
        return ready_tasks[0]

    @override
    def select_vm(self, task: TaskDto, vms: list[VmDto]) -> VmDto:
        """Schedule the task on the next VM in the list."""
        while not self.is_vm_suitable(vms[self.vm_index], task):
            self.vm_index = (self.vm_index + 1) % len(vms)

        selected_vm = vms[self.vm_index]
        # Move the cursor to the next VM
        self.vm_index = (self.vm_index + 1) % len(vms)

        return selected_vm
