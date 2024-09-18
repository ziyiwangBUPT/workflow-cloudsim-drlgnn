import tyro
import random
import json
import dataclasses

import numpy as np

from dataset_generator.core.gen_dataset import generate_dataset


@dataclasses.dataclass
class Args:
    seed: int = 42
    """random seed"""
    host_count: int = 2
    """number of hosts"""
    vm_count: int = 4
    """number of VMs"""
    max_cores: int = 10
    """maximum number of cores per VM"""
    min_cpu_speed: int = 500
    """minimum CPU speed in MIPS"""
    max_cpu_speed: int = 5000
    """maximum CPU speed in MIPS"""
    workflow_count: int = 3
    """number of workflows"""
    dag_method: str = "gnp"
    """DAG generation method (pegasus, gnp)"""
    gnp_min_n: int = 1
    """minimum number of tasks per workflow (for G(n,p) method)"""
    gnp_max_n: int = 5
    """maximum number of tasks per workflow (for G(n,p) method)"""
    task_length_dist: str = "normal"
    """task length distribution (normal, uniform, left_skewed, right_skewed)"""
    min_task_length: int = 500
    """minimum task length"""
    max_task_length: int = 100_000
    """maximum task length"""
    arrival_rate: float = 3
    """arrival rate of workflows (per second)"""


def main(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = generate_dataset(
        host_count=args.host_count,
        vm_count=args.vm_count,
        max_cores=args.max_cores,
        min_cpu_speed_mips=args.min_cpu_speed,
        max_cpu_speed_mips=args.max_cpu_speed,
        workflow_count=args.workflow_count,
        dag_method=args.dag_method,
        gnp_min_n=args.gnp_min_n,
        gnp_max_n=args.gnp_max_n,
        task_length_dist=args.task_length_dist,
        min_task_length=args.min_task_length,
        max_task_length=args.max_task_length,
        arrival_rate=args.arrival_rate,
    )

    json_data = json.dumps(dataclasses.asdict(dataset))
    print(json_data)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
