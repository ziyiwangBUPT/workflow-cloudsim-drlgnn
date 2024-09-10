import click
import random
import json
import dataclasses

import numpy as np

from dataset_generator.core.gen_dataset import generate_dataset


@click.command()
@click.option("--seed", default=42, help="Random seed", type=int)
@click.option("--host_count", default=2, help="Number of hosts")
@click.option("--vm_count", default=4, help="Number of VMs")
@click.option("--max_cores", default=10, help="Maximum number of cores per VM")
@click.option("--min_cpu_speed_mips", default=500, help="Minimum CPU speed in MIPS")
@click.option("--max_cpu_speed_mips", default=5000, help="Maximum CPU speed in MIPS")
@click.option("--workflow_count", default=3, help="Number of workflows", type=int)
@click.option("--min_task_count", default=1, help="Minimum number of tasks per workflow", type=int)
@click.option("--max_task_count", default=5, help="Maximum number of tasks per workflow", type=int)
@click.option("--min_task_length", default=500, help="Minimum task length", type=int)
@click.option("--max_task_length", default=100_000, help="Maximum task length", type=int)
@click.option("--arrival_rate", default=3, help="Arrival rate of workflows", type=int)
def main(
    seed: int,
    host_count: int,
    vm_count: int,
    max_cores: int,
    min_cpu_speed_mips: int,
    max_cpu_speed_mips: int,
    workflow_count: int,
    min_task_count: int,
    max_task_count: int,
    min_task_length: int,
    max_task_length: int,
    arrival_rate: int,
):
    random.seed(seed)
    np.random.seed(seed)

    dataset = generate_dataset(
        host_count=host_count,
        vm_count=vm_count,
        max_cores=max_cores,
        min_cpu_speed_mips=min_cpu_speed_mips,
        max_cpu_speed_mips=max_cpu_speed_mips,
        workflow_count=workflow_count,
        min_task_count=min_task_count,
        max_task_count=max_task_count,
        min_task_length=min_task_length,
        max_task_length=max_task_length,
        arrival_rate=arrival_rate,
    )

    json_data = json.dumps(dataclasses.asdict(dataset))
    print(json_data)


if __name__ == "__main__":
    main()
