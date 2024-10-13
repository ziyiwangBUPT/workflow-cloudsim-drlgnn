from gym_simulator.core.types import TaskDto, VmAssignmentDto, VmDto


def makespan_calculator(tasks: list[TaskDto], vms: list[VmDto], assignments: list[VmAssignmentDto]):
    """
    Calculate the makespan of the schedule.

    Makespan is the total time taken to complete all the tasks.
    """
    vm_completion_times = {vm.id: 0.0 for vm in vms}
    task_min_start_times = {(task.workflow_id, task.id): 0.0 for task in tasks}

    task_map = {(task.workflow_id, task.id): task for task in tasks}
    vm_map = {vm.id: vm for vm in vms}

    for assignment in assignments:
        task = task_map[(assignment.workflow_id, assignment.task_id)]
        vm = vm_map[assignment.vm_id]

        start_time = max(vm_completion_times[vm.id], task_min_start_times[(task.workflow_id, task.id)])
        computation_time = task.length / vm.cpu_speed_mips
        completion_time = start_time + computation_time

        vm_completion_times[vm.id] = completion_time
        for child_id in task.child_ids:
            child_task_id = (task.workflow_id, child_id)
            task_min_start_times[child_task_id] = max(completion_time, task_min_start_times[child_task_id])

    makespan = max(vm_completion_times.values())
    return makespan
