from collections import defaultdict

from matplotlib import pyplot as plt
import networkx as nx
from heft import heft, gantt
import numpy as np

from dataset_generator.visualizers.utils import draw_agraph
from gym_simulator.algorithms.base import BaseScheduler
from gym_simulator.algorithms.types import TaskDto, VmAssignmentDto, VmDto


ScheduleType = dict[int, list[heft.ScheduleEvent]]


class HeftOneScheduler(BaseScheduler):
    """
    Implementation of the HEFT scheduling algorithm.
    This adds a dummy start/end nodes and combine all workflows into one to be scheduled once.

    HEFT is a scheduling algorithm that uses a combination of task-level and workflow-level scheduling.
    Following implementation uses library: https://github.com/mackncheesiest/heft
    """

    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        tasks_in_workflow = [0] * (max(_task.workflow_id for _task in tasks) + 1)
        for task in tasks:
            tasks_in_workflow[task.workflow_id] += 1

        def _map(workflow_id: int, task_id: int) -> int:
            mapped_task_id = sum(tasks_in_workflow[:workflow_id]) + task_id
            mapped_task_id += 1  # Dummy start task has ID 0 (offset 1)
            return mapped_task_id

        def _unmap(mapped_task_id: int) -> tuple[int, int]:
            mapped_task_id -= 1  # Remove offset by dummy start task
            for workflow_id in range(len(tasks_in_workflow)):
                tasks_upto_now = sum(tasks_in_workflow[:workflow_id])
                current_workflow_tasks = tasks_in_workflow[workflow_id]
                if tasks_upto_now <= mapped_task_id < tasks_upto_now + current_workflow_tasks:
                    return (workflow_id, mapped_task_id - tasks_upto_now)

            raise Exception("Out of range")

        dummy_start_task = TaskDto(
            id=0,
            workflow_id=0,
            length=0,
            req_memory_mb=0,
            child_ids=[_map(_task.workflow_id, 0) for _task in tasks if _task.id == 0],
        )
        dummy_end_task = TaskDto(
            id=sum(tasks_in_workflow) + 1,
            workflow_id=0,
            length=0,
            req_memory_mb=0,
            child_ids=[],
        )

        mapped_tasks: list[TaskDto] = [dummy_start_task]
        for task in tasks:
            mapped_child_ids = [_map(task.workflow_id, child_id) for child_id in task.child_ids]
            if not mapped_child_ids:
                mapped_child_ids = [dummy_end_task.id]

            mapped_task = TaskDto(
                id=_map(task.workflow_id, task.id),
                workflow_id=0,
                length=task.length,
                req_memory_mb=task.req_memory_mb,
                child_ids=mapped_child_ids,
            )
            mapped_tasks.append(mapped_task)
        mapped_tasks.append(dummy_end_task)

        assignments: list[tuple[np.float64, VmAssignmentDto]] = []
        sched = self.schedule_workflow(mapped_tasks, vms)
        for vm_id, events in sched.items():
            for event in events:
                if event.task == 0:  # Dummy start
                    continue
                if event.task > sum(tasks_in_workflow):
                    continue
                workflow_id, task_id = _unmap(event.task)
                assignments.append((event.start, VmAssignmentDto(vm_id, workflow_id, task_id)))

        # Sort assignments by start time (make sure the order is correct)
        assignments.sort(key=lambda x: x[0])
        return [assignment for _, assignment in assignments]

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
