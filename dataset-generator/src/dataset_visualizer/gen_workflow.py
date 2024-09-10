import click
import random

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from dataset_generator.gen_workflow import generate_workflows
from dataset_visualizer.utils.plot_graphs import plot_workflows, draw_agraph


@click.command()
@click.option("--seed", default=0, help="Random seed", type=int)
@click.option("--workflow_count", default=5, help="Number of workflows", type=int)
@click.option("--min_task_count", default=1, help="Minimum number of tasks per workflow", type=int)
@click.option("--max_task_count", default=10, help="Maximum number of tasks per workflow", type=int)
@click.option("--min_task_length", default=500, help="Minimum task length", type=int)
@click.option("--max_task_length", default=100_000, help="Maximum task length", type=int)
@click.option("--max_req_cores", default=2, help="Maximum number of required cores per task", type=int)
@click.option("--arrival_rate", default=3, help="Arrival rate of workflows", type=int)
def main(
    seed: int,
    workflow_count: int,
    min_task_count: int,
    max_task_count: int,
    min_task_length: int,
    max_task_length: int,
    max_req_cores: int,
    arrival_rate: int,
):
    random.seed(seed)
    np.random.seed(seed)

    workflows = generate_workflows(
        workflow_count=workflow_count,
        min_task_count=min_task_count,
        max_task_count=max_task_count,
        min_task_length=min_task_length,
        max_task_length=max_task_length,
        max_req_cores=max_req_cores,
        arrival_rate=arrival_rate,
    )

    _, ax = plt.subplots()
    G: nx.DiGraph = nx.DiGraph()
    A = plot_workflows(G, workflows)
    draw_agraph(ax, A)
    plt.savefig("tmp/gen_workflows.png")


if __name__ == "__main__":
    main()
