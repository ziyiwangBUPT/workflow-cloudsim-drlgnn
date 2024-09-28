from abc import ABC

from gym_simulator.algorithms.base_heuristic import BaseHeuristicScheduler
from gym_simulator.algorithms.types import TaskDto, VmDto


class RandomMinScheduler(BaseHeuristicScheduler, ABC):
    def schedule_next(self, task: TaskDto, vms: list[VmDto]) -> VmDto:
        assert self.est_vm_completion_times is not None
        assert self.est_task_min_start_times is not None

        # Select the best VM by comparing the completion times
        best_vm = None
        best_vm_completion_time = float("inf")
        for vm in vms:
            if not self.is_vm_suitable(vm, task):
                continue

            completion_time = (
                max(self.est_vm_completion_times[self.vid(vm)], self.est_task_min_start_times[self.tid(task)])
                + task.length / vm.cpu_speed_mips
            )
            if best_vm_completion_time > completion_time:
                best_vm = vm
                best_vm_completion_time = completion_time

        if best_vm is None:
            raise Exception("No VM found for task")

        return best_vm
