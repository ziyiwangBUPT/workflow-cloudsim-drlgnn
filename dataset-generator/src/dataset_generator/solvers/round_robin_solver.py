import collections

from dataset_generator.core.models import VmAssignment, Task, Vm, Workflow


def solve_round_robin(workflows: list[Workflow], vms: list[Vm]) -> list[VmAssignment]:
    """
    Solve the VM assignment problem using a round-robin.
    """

    def is_assignable(task: Task, vm: Vm):
        return vm.cores >= task.req_cores

    def runtime(task_length: int, vm_speed: int):
        return int(task_length / vm_speed)

    # Initialize the ready set with the start tasks
    ready: collections.deque[tuple[int, int]] = collections.deque()
    for workflow in workflows:
        ready.append((workflow.id, 0))

    workflow_dict = {workflow.id: workflow for workflow in workflows}
    task_dict = {(workflow.id, task.id): task for workflow in workflows for task in workflow.tasks}
    assigned_task_dict: dict[tuple[int, int], VmAssignment] = {}
    vm_available_time = {vm.id: 0 for vm in vms}

    index = 0
    while ready:
        workflow_id, task_id = ready.popleft()
        workflow = workflow_dict[workflow_id]
        task = task_dict[(workflow_id, task_id)]

        best_vm = vms[index % len(vms)]
        index += 1
        while not is_assignable(task, best_vm):
            best_vm = vms[index % len(vms)]
            index += 1

        my_dependencies = [dep.id for dep in workflow.tasks if task_id in dep.child_ids]
        dependency_end_times = [assigned_task_dict[(workflow_id, dep)].end for dep in my_dependencies]
        start_time = max([vm_available_time[best_vm.id], *dependency_end_times])
        end_time = start_time + runtime(task.length, best_vm.cpu_speed_mips)

        vm_available_time[best_vm.id] = end_time
        assigned_task = VmAssignment(workflow_id, task_id, best_vm.id, start_time, end_time)
        assigned_task_dict[(workflow_id, task.id)] = assigned_task

        # Add the task to the ready set if all dependencies are done
        for child_id in task.child_ids:
            child_dependencies = [dep.id for dep in workflow.tasks if child_id in dep.child_ids]
            if all((workflow_id, dep) in assigned_task_dict for dep in child_dependencies):
                ready.append((workflow_id, child_id))

    return list(assigned_task_dict.values())
