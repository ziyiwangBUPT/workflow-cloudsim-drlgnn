import networkx as nx
from heft import heft
import numpy as np

from gym_simulator.algorithms.base import BaseScheduler
from gym_simulator.core.types import TaskDto, VmAssignmentDto, VmDto
from gym_simulator.utils.task_mapper import TaskMapper


ScheduleType = dict[int, list[heft.ScheduleEvent]]


class HeftOneScheduler(BaseScheduler):
    """
    Implementation of the HEFT scheduling algorithm.
    This adds a dummy start/end nodes and combine all workflows into one to be scheduled once.

    HEFT is a scheduling algorithm that uses a combination of task-level and workflow-level scheduling.
    Following implementation uses library: https://github.com/mackncheesiest/heft
    """

    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        assignments = self.schedule_with_time(tasks, vms)
        # Sort assignments by start time (make sure the order is correct)
        assignments.sort(key=lambda x: x[0])
        return [assignment for _, __, assignment in assignments]

    def schedule_with_time(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[tuple[float, float, VmAssignmentDto]]:
        task_mapper = TaskMapper(tasks)
        mapped_tasks = task_mapper.map_tasks()

        assignments: list[tuple[float, float, VmAssignmentDto]] = []
        sched = self.schedule_workflow(mapped_tasks, vms)
        for vm_id, events in sched.items():
            for event in events:
                if event.task == task_mapper.dummy_start_task_id():
                    continue
                if event.task == task_mapper.dummy_end_task_id():
                    continue
                workflow_id, task_id = task_mapper.unmap_id(event.task)
                assignments.append((float(event.start), float(event.end), VmAssignmentDto(vm_id, workflow_id, task_id)))

        return assignments

    def schedule_workflow(self, tasks: list[TaskDto], vms: list[VmDto]) -> ScheduleType:
        total_tasks = len(tasks)
        total_vms = len(vms)

        # Computational cost between tasks and vms
        comp_matrix = np.zeros((total_tasks, total_vms))
        for i in range(total_tasks):
            for j in range(total_vms):
                if not self.is_vm_suitable(vms[j], tasks[i]):
                    comp_matrix[i, j] = np.inf
                else:
                    comp_matrix[i, j] = tasks[i].length / vms[j].cpu_speed_mips
        # Communication cost between tasks - 0 if tasks are on the same VM, 1 otherwise
        comm_matrix = 1 - np.eye(total_vms)
        # Communication startup for VMs - 0 for all VMs
        comm_startup = np.zeros(total_vms)

        dag: nx.DiGraph = nx.DiGraph()
        for task in tasks:
            dag.add_node(task.id)
            for child_id in task.child_ids:
                dag.add_edge(task.id, child_id, weight=1)

        sched, _, _ = heft.schedule_dag(dag, comp_matrix, comm_matrix, comm_startup)
        assert sched is not None, "HEFT scheduling failed"
        return sched
