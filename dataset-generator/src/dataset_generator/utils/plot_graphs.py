import io

import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv

from dataset_generator.core.models import Workflow, Vm, VmAssignment


def color(id: int) -> str:
    color_map = ["#C96868", "#FADFA1", "#7EACB5", "#E6B9A6", "#939185", "#FFF078", "#939185"]
    return color_map[id % len(color_map)]


def get_node_id(workflow_id: int, task_id: int) -> str:
    return f"{workflow_id}-{task_id}"


def draw_agraph(ax: plt.Axes, A: pgv.AGraph, prog_args: str = ""):
    """
    Draw the provided AGraph on the provided Axes.
    """

    A.layout(prog="dot", args=prog_args)
    buffer = io.BytesIO()
    buffer.write(A.draw(format="png"))
    buffer.seek(0)
    ax.imshow(plt.imread(buffer))
    ax.axis("off")


def save_agraph(A: pgv.AGraph, path: str, prog_args: str = ""):
    """
    Save the provided AGraph to the provided path.
    """

    A.layout(prog="dot", args=prog_args)
    A.draw(path)


def plot_workflows(G: nx.DiGraph, workflows: list[Workflow]) -> pgv.AGraph:
    """
    Plot the workflows on the provided DiGraph.
    """

    for workflow in workflows:
        for task in workflow.tasks:
            node_id = get_node_id(workflow.id, task.id)
            node_label = f"{task.id} ({workflow.id})\n{task.length} MI\n{task.req_cores} vCPU"
            node_color = color(workflow.id)
            G.add_node(node_id, label=node_label, fillcolor=node_color, style="filled")
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
        start_times[node_id] = assignment.start

    # Add edges between tasks executed on the same VM
    for vm_id in vm_nodes:
        vm_nodes[vm_id].sort(key=lambda node_id: start_times[node_id])
        for i in range(1, len(vm_nodes[vm_id])):
            G.add_edge(vm_nodes[vm_id][i - 1], vm_nodes[vm_id][i], color="lightgray")

    # Add nodes and edges for tasks
    A = plot_workflows(G, workflows)

    # Add subgraphs for each VM
    for vm in vms:
        if vm_nodes[vm.id]:
            label = f"VM {vm.id}\n{int(vm.cpu_speed_mips)} MIPS\n{int(vm.cores)} vCPU"
            A.add_subgraph(vm_nodes[vm.id], name=f"cluster_{vm.id}", style="dashed", label=label)

    return A
