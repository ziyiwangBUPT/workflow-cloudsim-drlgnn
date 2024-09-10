import io

import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv

from dataset_generator.core.models import Workflow, VmAssignment, Vm
from dataset_generator.utils.plot_graphs import color, plot_workflows


def print_solution(workflows: list[Workflow], result: list[VmAssignment]):
    makespan_start = {workflow.id: float("inf") for workflow in workflows}
    makespan_end = {workflow.id: float("-inf") for workflow in workflows}

    for assignment in result:
        workflow_id = assignment.workflow_id
        makespan_start[workflow_id] = min(makespan_start[workflow_id], assignment.start)
        makespan_end[workflow_id] = max(makespan_end[workflow_id], assignment.end)
        print(assignment)
    total_makespan = sum(makespan_end[workflow_id] - makespan_start[workflow_id] for workflow_id in makespan_start)

    print()
    print(f"Total makespan: {total_makespan}")


def plot_gantt_chart(ax: plt.Axes, workflows: list[Workflow], vms: list[Vm], result: list[VmAssignment]):
    result_map: dict[tuple[int, int], VmAssignment] = {
        (assignment.workflow_id, assignment.task_id): assignment for assignment in result
    }

    for workflow in workflows:
        for task in workflow.tasks:
            assigned_task = result_map[(workflow.id, task.id)]
            ax.broken_barh(
                [(assigned_task.start, assigned_task.end - assigned_task.start)],
                (int(assigned_task.vm_id) - 0.3, 0.6),
                color=color(workflow.id),
                edgecolor="black",
                linewidth=0.5,
            )

    ax.set_yticks(range(len(vms)))
    ax.set_yticklabels([f"VM {vm.id}\n{int(vm.cpu_speed_mips)} MIPS\n{int(vm.cores)} vCPU" for vm in vms])
    ax.set_xlabel("Time")
