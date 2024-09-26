from gym_simulator.algorithms.types import VmDto, TaskDto, VmAssignmentDto


def round_robin(tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
    vm_index = 0

    assignments: list[VmAssignmentDto] = []
    for task in tasks:
        vm_index = (vm_index + 1) % len(vms)
        while vms[vm_index].cores < task.req_cores:
            vm_index = (vm_index + 1) % len(vms)

        assignments.append(VmAssignmentDto(vms[vm_index].id, task.workflow_id, task.id))

    return assignments


def best_fit(tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
    assignments: list[VmAssignmentDto] = []
    for task in tasks:
        best_vm_index = None
        for vm_index, vm in enumerate(vms):
            if vm.cores >= task.req_cores:
                if best_vm_index is None or vms[best_vm_index].cores > vm.cores:
                    best_vm_index = vm_index

        if best_vm_index is None:
            raise Exception("No VM found for task")

        assignments.append(VmAssignmentDto(vms[best_vm_index].id, task.workflow_id, task.id))

    return assignments
