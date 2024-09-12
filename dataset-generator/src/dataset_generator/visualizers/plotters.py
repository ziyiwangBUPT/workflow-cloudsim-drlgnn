import io

import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv

from dataset_generator.core.models import Workflow, VmAssignment, Vm


# -------------------------------------------------------------------------------------------------
# Helper functions


def color(id: int) -> str:
    color_map = ["#C96868", "#FADFA1", "#7EACB5", "#E6B9A6", "#939185", "#FFF078", "#939185"]
    return color_map[id % len(color_map)]


def get_node_id(workflow_id: int, task_id: int) -> str:
    return f"{workflow_id}-{task_id}"


# -------------------------------------------------------------------------------------------------
# Graphing functions for Workflow and Execution graphs


def plot_workflow_graphs(G: nx.DiGraph, workflows: list[Workflow], **node_attr) -> pgv.AGraph:
    """
    Plot the workflows on the provided DiGraph.
    """

    for workflow in workflows:
        for task in workflow.tasks:
            node_id = get_node_id(workflow.id, task.id)
            node_label = f"W{workflow.id} T{task.id}\n{task.length} MI\n{task.req_cores} vCPU"
            node_color = color(workflow.id)
            G.add_node(node_id, label=node_label, fillcolor=node_color, style="filled", fontname="Arial", **node_attr)
            for child_id in task.child_ids:
                G.add_edge(node_id, get_node_id(workflow.id, child_id), color=node_color)

    return nx.nx_agraph.to_agraph(G)


def plot_execution_graph(G: nx.DiGraph, workflows: list[Workflow], vms: list[Vm], result: list[VmAssignment]):
    """
    Plot the execution graph of the provided workflows on the provided DiGraph.
    """

    # Assignments
    vm_nodes: dict[int, list[str]] = {vm.id: [] for vm in vms}
    start_times: dict[str, int] = {}
    for assignment in result:
        node_id = get_node_id(assignment.workflow_id, assignment.task_id)
        vm_nodes[assignment.vm_id].append(node_id)
        start_times[node_id] = assignment.start_time

    # Add edges between tasks executed on the same VM
    for vm_id in vm_nodes:
        vm_nodes[vm_id].sort(key=lambda node_id: start_times[node_id])
        for i in range(1, len(vm_nodes[vm_id])):
            G.add_edge(vm_nodes[vm_id][i - 1], vm_nodes[vm_id][i], color="lightgray")

    # Add nodes and edges for tasks
    A = plot_workflow_graphs(G, workflows, shape="box")

    # Add subgraphs for each VM
    for vm in vms:
        if vm_nodes[vm.id]:
            label = f"VM {vm.id}\n{int(vm.cpu_speed_mips)} MIPS\n{int(vm.cores)} vCPU"
            A.add_subgraph(vm_nodes[vm.id], name=f"cluster_{vm.id}", style="dashed", fontname="Arial", label=label)

    return A


# -------------------------------------------------------------------------------------------------
# Graphing functions for Gantt chart


def plot_gantt_chart(ax: plt.Axes, workflows: list[Workflow], vms: list[Vm], result: list[VmAssignment]):
    result_map: dict[tuple[int, int], VmAssignment] = {
        (assignment.workflow_id, assignment.task_id): assignment for assignment in result
    }

    for workflow in workflows:
        for task in workflow.tasks:
            assigned_task = result_map[(workflow.id, task.id)]
            ax.broken_barh(
                [(assigned_task.start_time, assigned_task.end_time - assigned_task.start_time)],
                (int(assigned_task.vm_id) - 0.3, 0.6),
                color=color(workflow.id),
                edgecolor="black",
                linewidth=0.5,
            )
            # Put text in middle
            ax.text(
                x=assigned_task.start_time + (assigned_task.end_time - assigned_task.start_time) / 2,
                y=int(assigned_task.vm_id),
                s=f"W{workflow.id}\nT{task.id}",
                ha="center",
                va="center",
            )

    ax.set_yticks(range(len(vms)))
    ax.set_yticklabels([f"VM {vm.id}\n{int(vm.cpu_speed_mips)} MIPS\n{int(vm.cores)} vCPU" for vm in vms])
    ax.set_xlabel("Time")
