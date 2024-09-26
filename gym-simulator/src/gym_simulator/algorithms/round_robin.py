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
