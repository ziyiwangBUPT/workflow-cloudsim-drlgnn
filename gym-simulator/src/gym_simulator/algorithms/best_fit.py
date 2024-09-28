from gym_simulator.algorithms.base_heuristic import BaseHeuristicScheduler
from gym_simulator.algorithms.types import TaskDto, VmDto


class BestFitScheduler(BaseHeuristicScheduler):
    def schedule_next(self, task: TaskDto, vms: list[VmDto]) -> VmDto:
        assert self.est_vm_completion_times is not None
        assert self.est_task_min_start_times is not None

        best_vm = None
        best_vm_allocation = -float("inf")
        for vm in vms:
            if not self.is_vm_suitable(vm, task):
                continue
            vm_allocation = task.req_memory_mb / vm.memory_mb
            assert 0 <= vm_allocation <= 1, f"Invalid VM allocation: {vm_allocation}"

            # If the current VM has a better fit, update the best VM
            if vm_allocation > best_vm_allocation:
                best_vm = vm
                best_vm_allocation = vm_allocation

            # If the current VM has the same memory, check the estimated completion time
            elif vm_allocation == best_vm_allocation:
                assert best_vm is not None
                if self.est_vm_completion_times[self.vid(vm)] < self.est_vm_completion_times[self.vid(best_vm)]:
                    best_vm = vm
                    best_vm_allocation = vm_allocation

        if best_vm is None:
            raise Exception("No VM found for task")

        return best_vm
