from gym_simulator.algorithms.base_heuristic import BaseHeuristicScheduler
from gym_simulator.algorithms.types import TaskDto, VmDto


class RoundRobinScheduler(BaseHeuristicScheduler):
    vm_index: int = 0

    def schedule_next(self, task: TaskDto, vms: list[VmDto]) -> VmDto:
        while not self.is_vm_suitable(vms[self.vm_index], task):
            self.vm_index = (self.vm_index + 1) % len(vms)

        selected_vm = vms[self.vm_index]
        # Move the cursor to the next VM
        self.vm_index = (self.vm_index + 1) % len(vms)

        return selected_vm
