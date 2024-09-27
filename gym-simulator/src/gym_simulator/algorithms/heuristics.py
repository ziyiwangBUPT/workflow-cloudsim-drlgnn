from abc import ABC, abstractmethod

from collections import deque

from gym_simulator.algorithms.types import VmDto, TaskDto, VmAssignmentDto


def is_vm_suitable(vm: VmDto, task: TaskDto) -> bool:
    return vm.memory_mb >= task.req_memory_mb


class AbstractHeuristicScheduler(ABC):
    def __init__(self, tasks: list[TaskDto], vms: list[VmDto]):
        self.tasks = tasks
        self.vms = vms

        self.task_map = {(task.workflow_id, task.id): task for task in tasks}
        self.ready_tasks = deque([(task.workflow_id, task.id) for task in tasks if task.id == 0])
        self.processed_tasks: set[tuple[int, int]] = set()

    @abstractmethod
    def schedule(self) -> list[VmAssignmentDto]:
        raise NotImplementedError

    def add_child_tasks(self, task: TaskDto):
        """Add child tasks to the ready queue if all parents are processed."""
        for child_id in task.child_ids:
            workflow_child_id = (task.workflow_id, child_id)
            all_parents_processed = all(
                (parent.workflow_id, parent.id) in self.processed_tasks
                for parent in self.tasks
                if parent.workflow_id == task.workflow_id and child_id in parent.child_ids
            )
            if all_parents_processed and workflow_child_id not in self.ready_tasks:
                self.ready_tasks.append(workflow_child_id)


class RoundRobinScheduler(AbstractHeuristicScheduler):
    def schedule(self) -> list[VmAssignmentDto]:
        assignments: list[VmAssignmentDto] = []
        vm_index = 0

        while self.ready_tasks:
            workflow_task_id = self.ready_tasks.popleft()
            self.processed_tasks.add(workflow_task_id)
            task = self.task_map[workflow_task_id]

            while not is_vm_suitable(self.vms[vm_index], task):
                vm_index = (vm_index + 1) % len(self.vms)

            assignments.append(VmAssignmentDto(self.vms[vm_index].id, task.workflow_id, task.id))
            vm_index = (vm_index + 1) % len(self.vms)

            self.add_child_tasks(task)
        return assignments


class BestFitScheduler(AbstractHeuristicScheduler):
    def schedule(self) -> list[VmAssignmentDto]:
        assignments: list[VmAssignmentDto] = []
        est_vm_completion_times = [0] * len(self.vms)
        est_task_start_times = {(task.workflow_id, task.id): 0 for task in self.tasks}

        while self.ready_tasks:
            workflow_task_id = self.ready_tasks.popleft()
            self.processed_tasks.add(workflow_task_id)
            task = self.task_map[workflow_task_id]

            best_vm_index = None
            best_vm_allocation = float("inf")
            for vm_index, vm in enumerate(self.vms):
                if not is_vm_suitable(vm, task):
                    continue
                vm_allocation = task.req_memory_mb / vm.memory_mb
                assert 0 <= vm_allocation <= 1, f"Invalid VM allocation: {vm_allocation}"

                # If the current VM has a better fit, update the best VM
                if best_vm_index is None or vm_allocation > best_vm_allocation:
                    best_vm_index = vm_index
                    best_vm_allocation = vm_allocation

                # If the current VM has the same memory, check the estimated completion time
                elif vm_allocation == best_vm_allocation:
                    if est_vm_completion_times[vm_index] < est_vm_completion_times[best_vm_index]:
                        best_vm_index = vm_index
                        best_vm_allocation = vm_allocation

            if best_vm_index is None:
                raise Exception("No VM found for task")

            assignments.append(VmAssignmentDto(self.vms[best_vm_index].id, task.workflow_id, task.id))

            # Update the estimated completion times and task start times
            est_process_time = task.length / self.vms[best_vm_index].cpu_speed_mips
            est_end_time = (
                max(est_vm_completion_times[best_vm_index], est_task_start_times[workflow_task_id]) + est_process_time
            )
            est_vm_completion_times[best_vm_index] = est_end_time
            for child_id in task.child_ids:
                est_task_start_times[(task.workflow_id, child_id)] = est_end_time
            self.add_child_tasks(task)

        return assignments
