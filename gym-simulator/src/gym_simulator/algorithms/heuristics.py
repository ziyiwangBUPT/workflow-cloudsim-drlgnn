from collections import deque

from gym_simulator.algorithms.types import VmDto, TaskDto, VmAssignmentDto


def is_vm_suitable(vm: VmDto, task: TaskDto) -> bool:
    return vm.cores >= task.req_cores


def round_robin(tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
    vm_index = 0

    assignments: list[VmAssignmentDto] = []
    for task in tasks:
        vm_index = (vm_index + 1) % len(vms)
        while not is_vm_suitable(vms[vm_index], task):
            vm_index = (vm_index + 1) % len(vms)

        assignments.append(VmAssignmentDto(vms[vm_index].id, task.workflow_id, task.id))

    return assignments


def best_fit(tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
    assignments: list[VmAssignmentDto] = []
    for task in tasks:
        best_vm_index = None
        for vm_index, vm in enumerate(vms):
            if is_vm_suitable(vm, task):
                if best_vm_index is None or vms[best_vm_index].cores > vm.cores:
                    best_vm_index = vm_index

        if best_vm_index is None:
            raise Exception("No VM found for task")

        assignments.append(VmAssignmentDto(vms[best_vm_index].id, task.workflow_id, task.id))

    return assignments


def min_min(tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
    estimated_completion_times_of_vms = [0] * len(vms)
    assignments: list[VmAssignmentDto] = []
    for task in tasks:
        best_vm_index = None
        for vm_index, vm in enumerate(vms):
            if is_vm_suitable(vm, task):
                estimated_completion_time = estimated_completion_times_of_vms[vm_index] + task.length
                if (
                    best_vm_index is None
                    or estimated_completion_time < estimated_completion_times_of_vms[best_vm_index]
                ):
                    best_vm_index = vm_index

        if best_vm_index is None:
            raise Exception("No VM found for task")

        assignments.append(VmAssignmentDto(vms[best_vm_index].id, task.workflow_id, task.id))
        estimated_completion_times_of_vms[best_vm_index] += task.length

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
