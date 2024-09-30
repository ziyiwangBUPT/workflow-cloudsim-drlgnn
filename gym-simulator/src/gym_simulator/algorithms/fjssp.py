from collections import defaultdict
from typing import override

from gym_simulator.algorithms.base_ready_queue import BaseReadyQueueScheduler
from gym_simulator.core.types import TaskDto, VmDto


class FjsspScheduler(BaseReadyQueueScheduler):
    task_select_algo: str
    vm_select_algo: str

    def __init__(self, task_select_algo: str, vm_select_algo: str):
        self.task_select_algo = task_select_algo
        self.vm_select_algo = vm_select_algo

    @override
    def select_task(self, ready_tasks: list[TaskDto]) -> TaskDto:
        assert self.task_map is not None
        pending_tasks = [task for task_id, task in self.task_map.items() if self.is_pending(task_id)]
        if self.task_select_algo == "fifo":
            return self.select_task_fifo(ready_tasks)
        elif self.task_select_algo == "mopnr":
            return self.select_task_mopnr(ready_tasks, pending_tasks)
        elif self.task_select_algo == "lwkr":
            return self.select_task_lwkr(ready_tasks, pending_tasks)
        elif self.task_select_algo == "mwkr":
            return self.select_task_mwkr(ready_tasks, pending_tasks)
        else:
            raise ValueError(f"Unknown task selection algorithm: {self.task_select_algo}")

    @override
    def select_vm(self, task: TaskDto, vms: list[VmDto]) -> VmDto:
        if self.vm_select_algo == "spt":
            return self.select_vm_spt(task, vms)
        elif self.vm_select_algo == "eet":
            return self.select_vm_eet(task, vms)
        else:
            raise ValueError(f"Unknown VM selection algorithm: {self.vm_select_algo}")

    # Task selection algorithms ------------------------------------------------

    def select_task_fifo(self, ready_tasks: list[TaskDto]) -> TaskDto:
        """Select task based on First In First Out (FIFO)"""
        return ready_tasks[0]

    def select_task_mopnr(self, ready_tasks: list[TaskDto], pending_tasks: list[TaskDto]) -> TaskDto:
        """Select task based on Most Operation Number Remaining (MOPNR)"""
        remaining_tasks = ready_tasks + pending_tasks
        grouped_remaining_tasks = self._group_to_workflows(remaining_tasks)

        selected_task_op_num_rem = -1
        selected_task = None
        for task in ready_tasks:
            my_workflow_tasks = grouped_remaining_tasks[task.workflow_id]
            op_num_rem = len(my_workflow_tasks)
            if op_num_rem > selected_task_op_num_rem:
                selected_task_op_num_rem = op_num_rem
                selected_task = task

        assert selected_task is not None
        return selected_task

    def select_task_lwkr(self, ready_tasks: list[TaskDto], pending_tasks: list[TaskDto]) -> TaskDto:
        """Select task based on Least Work Remaining (LWKR)"""
        remaining_tasks = ready_tasks + pending_tasks
        grouped_remaining_tasks = self._group_to_workflows(remaining_tasks)

        selected_task_work_rem = float("inf")
        selected_task = None
        for task in ready_tasks:
            my_workflow_tasks = grouped_remaining_tasks[task.workflow_id]
            work_rem = sum([_task.length for _task in my_workflow_tasks])
            if work_rem < selected_task_work_rem:
                selected_task_work_rem = work_rem
                selected_task = task

        assert selected_task is not None
        return selected_task

    def select_task_mwkr(self, ready_tasks: list[TaskDto], pending_tasks: list[TaskDto]) -> TaskDto:
        """Select task based on Most Work Remaining (MWKR)"""
        remaining_tasks = ready_tasks + pending_tasks
        grouped_remaining_tasks = self._group_to_workflows(remaining_tasks)

        selected_task_work_rem = -float("inf")
        selected_task = None
        for task in ready_tasks:
            my_workflow_tasks = grouped_remaining_tasks[task.workflow_id]
            work_rem = sum([_task.length for _task in my_workflow_tasks])
            if work_rem > selected_task_work_rem:
                selected_task_work_rem = work_rem
                selected_task = task

        assert selected_task is not None
        return selected_task

    # VM selection algorithms --------------------------------------------------

    def select_vm_spt(self, task: TaskDto, vms: list[VmDto]) -> VmDto:
        """Select VM based on Shortest Processing Time (SPT)"""
        selected_vm = None
        selected_vm_proc_time = float("inf")
        for vm in vms:
            if not self.is_vm_suitable(vm, task):
                continue
            proc_time = task.length / vm.cpu_speed_mips
            if proc_time < selected_vm_proc_time:
                selected_vm_proc_time = proc_time
                selected_vm = vm

        assert selected_vm is not None
        return selected_vm

    def select_vm_eet(self, task: TaskDto, vms: list[VmDto]) -> VmDto:
        """Select VM based on Earliest End Time (EET)"""
        assert self.est_vm_completion_times is not None
        assert self.est_task_min_start_times is not None

        selected_vm = None
        selected_vm_end_time = float("inf")
        for vm in vms:
            if not self.is_vm_suitable(vm, task):
                continue
            proc_time = task.length / vm.cpu_speed_mips
            task_can_start_time = self.est_task_min_start_times[(task.workflow_id, task.id)]
            vm_can_start_time = self.est_vm_completion_times[vm.id]
            start_time = max(task_can_start_time, vm_can_start_time)
            end_time = proc_time + start_time
            if end_time < selected_vm_end_time:
                selected_vm_end_time = end_time
                selected_vm = vm

        assert selected_vm is not None
        return selected_vm

    # Helper methods -----------------------------------------------------------

    def _group_to_workflows(self, tasks: list[TaskDto]) -> dict[int, list[TaskDto]]:
        workflows: defaultdict[int, list[TaskDto]] = defaultdict(list)
        for task in tasks:
            workflow_id = task.workflow_id
            workflows[workflow_id].append(task)

        return dict(workflows)
