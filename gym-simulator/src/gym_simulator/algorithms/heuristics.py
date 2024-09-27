from collections import deque

from gym_simulator.algorithms.types import VmDto, TaskDto, VmAssignmentDto


def is_vm_suitable(vm: VmDto, task: TaskDto) -> bool:
    return vm.memory_mb >= task.req_memory_mb


def round_robin(tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
    """
    Assigns tasks to Virtual Machines (VMs) using the Round Robin algorithm.

    The Round Robin algorithm cycles through available VMs and assigns each task
    to the next suitable VM in a circular manner, ensuring fair distribution of tasks
    across the VMs.
    """
    vm_index = 0

    assignments: list[VmAssignmentDto] = []
    for task in tasks:
        for vm_offset in range(len(vms)):
            check_vm_index = (vm_index + vm_offset) % len(vms)
            if is_vm_suitable(vms[check_vm_index], task):
                vm_index = check_vm_index
                break
        else:
            raise Exception("No VM found for task")

        assignments.append(VmAssignmentDto(vms[vm_index].id, task.workflow_id, task.id))
        vm_index = (vm_index + 1) % len(vms)

    return assignments


def best_fit(tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
    """
    Assigns tasks to Virtual Machines (VMs) using the Best Fit algorithm.

    The Best Fit algorithm evaluates available VMs to allocate each task to the VM
    that leaves the least amount of remaining capacity after the assignment.
    It prioritizes VMs based on their available resources (cores) and estimated
    completion times to minimize resource waste and balance the load across VMs.
    """

    est_vm_completion_times = [0] * len(vms)
    est_task_start_times = {(task.workflow_id, task.id): 0 for task in tasks}
    assignments: list[VmAssignmentDto] = []

    for task in tasks:
        best_vm_index = None
        best_vm_allocation = float("inf")
        for vm_index, vm in enumerate(vms):
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

        assignments.append(VmAssignmentDto(vms[best_vm_index].id, task.workflow_id, task.id))

        # Update the estimated completion times and task start times
        est_process_time = task.length / vms[best_vm_index].cpu_speed_mips
        est_end_time = (
            max(est_vm_completion_times[best_vm_index], est_task_start_times[(task.workflow_id, task.id)])
            + est_process_time
        )
        est_vm_completion_times[best_vm_index] = est_end_time
        for child_id in task.child_ids:
            est_task_start_times[(task.workflow_id, child_id)] = est_end_time

    return assignments


def min_min(tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
    estimated_completion_times_of_vms = [0] * len(vms)
    estimated_start_times_of_tasks = {(task.workflow_id, task.id): 0 for task in tasks}
    assignments: list[VmAssignmentDto] = []
    for task in tasks:
        best_vm_index = None
        best_vm_estimated_completion_time = float("inf")
        for vm_index, vm in enumerate(vms):
            if is_vm_suitable(vm, task):
                estimated_start_time = estimated_start_times_of_tasks[(task.workflow_id, task.id)]
                estimated_completion_time = (
                    max(estimated_completion_times_of_vms[vm_index], estimated_start_time) + task.length
                )
                if estimated_completion_time < best_vm_estimated_completion_time:
                    best_vm_index = vm_index
                    best_vm_estimated_completion_time = estimated_completion_time

        if best_vm_index is None:
            raise Exception("No VM found for task")

        assignments.append(VmAssignmentDto(vms[best_vm_index].id, task.workflow_id, task.id))
        estimated_completion_times_of_vms[best_vm_index] = best_vm_estimated_completion_time
        estimated_start_times_of_tasks[(task.workflow_id, task.id)] = best_vm_estimated_completion_time

    return assignments


def max_min(tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
    estimated_completion_times_of_vms = [0] * len(vms)
    assignments: list[VmAssignmentDto] = []
    for task in tasks:
        best_vm_index = None
        for vm_index, vm in enumerate(vms):
            if is_vm_suitable(vm, task):
                estimated_completion_time = estimated_completion_times_of_vms[vm_index] + task.length
                if (
                    best_vm_index is None
                    or estimated_completion_time > estimated_completion_times_of_vms[best_vm_index]
                ):
                    best_vm_index = vm_index

        if best_vm_index is None:
            raise Exception("No VM found for task")

        assignments.append(VmAssignmentDto(vms[best_vm_index].id, task.workflow_id, task.id))
        estimated_completion_times_of_vms[best_vm_index] += task.length

    return assignments
