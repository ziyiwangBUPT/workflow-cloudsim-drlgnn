import click
import json

import networkx as nx
import matplotlib.pyplot as plt

from dataset_generator.core.models import Dataset, Workflow, Vm, Host, Task
from dataset_generator.solvers.solver import solve
from dataset_generator.visualizers.utils import save_agraph
from dataset_generator.visualizers.plotters import plot_gantt_chart, plot_workflow_graphs, plot_execution_graph
from dataset_generator.visualizers.printers import print_solution


def workflow_from_json(data: dict) -> Workflow:
    tasks = [Task(**task) for task in data.pop("tasks")]
    return Workflow(tasks=tasks, **data)


def dataset_from_json(data: dict) -> Dataset:
    workflows = [workflow_from_json(workflow) for workflow in data.pop("workflows")]
    vms = [Vm(**vm) for vm in data.pop("vms")]
    hosts = [Host(**host) for host in data.pop("hosts")]
    return Dataset(workflows=workflows, vms=vms, hosts=hosts)


@click.command()
@click.option("--method", default="sat", help="Method to solve the dataset", type=click.Choice(["sat", "round_robin"]))
def main(method: str):
    dataset_str = input()
    dataset_dict = json.loads(dataset_str)
    dataset = dataset_from_json(dataset_dict)

    # Workflow graph
    _, ax = plt.subplots()
    G_w: nx.DiGraph = nx.DiGraph()
    A_w = plot_workflow_graphs(G_w, dataset.workflows)
    save_agraph(A_w, f"tmp/solve_datasets_{method}_workflows.png")

    # Solution
    result = solve(method, dataset)
    print_solution(dataset.workflows, result)

    # Execution graph
    _, ax = plt.subplots()
    G_e: nx.DiGraph = nx.DiGraph()
    A_e = plot_execution_graph(G_e, dataset.workflows, dataset.vms, result)
    save_agraph(A_e, f"tmp/solve_datasets_{method}_execution.png", dir_lr=True)

    # Gantt chart
    _, ax = plt.subplots()
    plot_gantt_chart(ax, dataset.workflows, dataset.vms, result)
    plt.savefig(f"tmp/solve_datasets_{method}_gantt.png")


if __name__ == "__main__":
    main()
