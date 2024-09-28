from multiprocessing import Process, Manager, Queue

from numpy import sort

from dataset_generator.core.models import Task, Vm, VmAssignment, Workflow
from dataset_generator.solvers.cp_sat_solver import solve_cp_sat
from gym_simulator.algorithms.base import BaseScheduler
from gym_simulator.algorithms.types import TaskDto, VmAssignmentDto, VmDto


def solve_cp_sat_queue(workflows: list[Workflow], vms: list[Vm], queue: Queue):
    assignments = solve_cp_sat(workflows, vms)
    queue.put(assignments)


class CpSatScheduler(BaseScheduler):
    """
    Implementation of the CP-SAT scheduling algorithm.

    CP-SAT is a scheduling algorithm that uses constraint programming to solve the scheduling problem.
    """

    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        grouped_task_objs: dict[int, list[Task]] = {}
        for task in tasks:
            if task.workflow_id not in grouped_task_objs:
                grouped_task_objs[task.workflow_id] = []
            grouped_task_objs[task.workflow_id].append(
                Task(
                    id=task.id,
                    workflow_id=task.workflow_id,
                    length=task.length,
                    req_memory_mb=task.req_memory_mb,
                    child_ids=task.child_ids,
                )
            )

        workflow_objs: list[Workflow] = []
        for workflow_id, task_objs in grouped_task_objs.items():
            workflow_objs.append(
                Workflow(
                    id=workflow_id,
                    tasks=task_objs,
                    arrival_time=0,  # Static scheduling (all tasks arrived t=0)
                )
            )

        vm_objs: list[Vm] = []
        for vm in vms:
            vm_objs.append(
                Vm(
                    id=vm.id,
                    host_id=0,  # Doesn't matter
                    cpu_speed_mips=int(vm.cpu_speed_mips),
                    memory_mb=vm.memory_mb,
                    disk_mb=500,  # Doesn't matter
                    bandwidth_mbps=500,  # Doesn't matter
                    vmm="Xen",  # Doesn't matter
                )
            )

        # Run the solver in a separate process and stop after 1 minute, also get the output
        with Manager() as manager:
            queue = manager.Queue()
            process = Process(target=solve_cp_sat_queue, args=(workflow_objs, vm_objs, queue))
            process.start()
            process.join(60)
            if process.is_alive():
                process.terminate()
                process.join()

            result = queue.get()

        # We have to sort since we lose start time and the assignments are supposed to be in temporal order
        assignment_objs: list[VmAssignment] = sorted(result, key=lambda t: t.start_time)

        assignments: list[VmAssignmentDto] = []
        for assignment_obj in assignment_objs:
            assignments.append(
                VmAssignmentDto(
                    vm_id=assignment_obj.vm_id,
                    workflow_id=assignment_obj.workflow_id,
                    task_id=assignment_obj.task_id,
                )
            )

        return assignments
