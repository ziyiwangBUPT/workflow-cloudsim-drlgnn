import tyro
import random
import json
import dataclasses

import numpy as np

from dataset_generator.core.gen_dataset import generate_dataset


@dataclasses.dataclass
class Args:
    seed: int = 42
    """Random seed"""

    host_count: int = 2
    """Number of hosts"""

    vm_count: int = 4
    """Number of VMs"""

    max_cores: int = 10
    """Maximum number of cores per VM"""

    min_cpu_speed: int = 500
    """Minimum CPU speed in MIPS"""

    max_cpu_speed: int = 5000
    """Maximum CPU speed in MIPS"""

    workflow_count: int = 3
    """Number of workflows"""

    min_task_count: int = 1
    """Minimum number of tasks per workflow"""

    max_task_count: int = 5
    """Maximum number of tasks per workflow"""

    task_length_dist: str = "normal"
    """Task length distribution"""

    min_task_length: int = 500
    """Minimum task length"""

    max_task_length: int = 100_000
    """Maximum task length"""

    arrival_rate: float = 3
    """Arrival rate of workflows (per second)"""


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
        min_task_count=args.min_task_count,
        max_task_count=args.max_task_count,
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
