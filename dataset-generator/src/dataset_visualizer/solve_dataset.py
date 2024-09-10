import click
import json
import random
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

from dataset_generator.models import Dataset, Workflow, Vm, Host, VmAllocation, Task
from dataset_visualizer.solvers.solver import solve
from dataset_visualizer.utils.plot_solutions import print_solution, plot_gantt_chart
from dataset_visualizer.utils.plot_graphs import plot_workflows, plot_execution_graph, save_agraph


def workflow_from_json(data: dict) -> Workflow:
    tasks = [Task(**task) for task in data.pop("tasks")]
    return Workflow(tasks=tasks, **data)


def dataset_from_json(data: dict) -> Dataset:
    workflows = [workflow_from_json(workflow) for workflow in data.pop("workflows")]
    vms = [Vm(**vm) for vm in data.pop("vms")]
    hosts = [Host(**host) for host in data.pop("hosts")]
    vm_allocations = [VmAllocation(**vm_allocation) for vm_allocation in data.pop("vm_allocations")]
    return Dataset(workflows=workflows, vms=vms, hosts=hosts, vm_allocations=vm_allocations)


@click.command()
@click.option("--method", default="sat", help="Method to solve the dataset", type=click.Choice(["sat", "round_robin"]))
def main(method: str):
    random.seed(0)
    np.random.seed(0)

    dataset_str = input()
    dataset_dict = json.loads(dataset_str)
    dataset = dataset_from_json(dataset_dict)

    # Workflow graph
    _, ax = plt.subplots()
    G_w: nx.DiGraph = nx.DiGraph()
    A_w = plot_workflows(G_w, dataset.workflows)
    save_agraph(A_w, f"tmp/solve_datasets_{method}_workflows.png")

    # Solution
    result = solve(method, dataset)
    print_solution(dataset.workflows, result)

    # Execution graph
    _, ax = plt.subplots()
    G_e: nx.DiGraph = nx.DiGraph()
    A_e = plot_execution_graph(G_e, dataset.workflows, dataset.vms, result)
    save_agraph(A_e, f"tmp/solve_datasets_{method}_execution.png", prog_args="-Grankdir=LR")

    # Gantt chart
    _, ax = plt.subplots()
    plot_gantt_chart(ax, dataset.workflows, dataset.vms, result)
    plt.savefig(f"tmp/solve_datasets_{method}_gantt.png")


if __name__ == "__main__":
    main()
