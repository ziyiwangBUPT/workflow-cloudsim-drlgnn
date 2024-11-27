import itertools
import time
from dataclasses import dataclass, field
from typing import Any

import tyro
from icecream import ic
from pandas import DataFrame
from tqdm import tqdm

from scheduler.config.settings import ALGORITHMS
from scheduler.dataset_generator.core.models import Solution
from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.viz_results.algorithms import algorithm_strategy
from scheduler.viz_results.algorithms.base import BaseScheduler
from scheduler.viz_results.simulation.gym_env import CloudSimGymEnvironment
from scheduler.viz_results.simulation.observation import SimEnvAction


@dataclass
class Args:
    simulator: str
    """path to the simulator JAR file"""
    seed: int
    """random seed"""
    buffer_size: int
    """size of the workflow scheduler buffer"""
    buffer_timeout: int
    """timeout of the workflow scheduler buffer"""
    export_csv: str
    """file to output the export CSV"""
    num_iterations: int = 10
    """number of iterations to evaluate"""
    dataset_args: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            host_count=10,
            vm_count=4,
            workflow_count=10,
            gnp_min_n=20,
            gnp_max_n=20,
            max_memory_gb=10,
            min_cpu_speed=500,
            max_cpu_speed=5000,
            min_task_length=500,
            max_task_length=100_000,
            task_arrival="static",
            dag_method="gnp",
        )
    )
    """the dataset"""


def run_algorithm(scheduler: BaseScheduler, seed: int, args: Args) -> tuple[float, float, float]:
    env = CloudSimGymEnvironment(args.simulator, args.dataset_args)

    total_scheduling_time: float = 0
    obs, info = env.reset(seed=seed)
    while True:
        scheduling_start_time = time.time()
        assignments = scheduler.schedule(obs.tasks, obs.vms)
        scheduling_end_time = time.time()
        total_scheduling_time += scheduling_end_time - scheduling_start_time
        obs, reward, terminated, truncated, info = env.step(SimEnvAction(assignments))
        if terminated or truncated:
            solution: Solution = info["solution"]
            energy_consumption: float = info.get("total_energy_consumption_j", 0)
            start_time = min([assignment.start_time for assignment in solution.vm_assignments])
            end_time = max([assignment.end_time for assignment in solution.vm_assignments])
            makespan = end_time - start_time
            return makespan, energy_consumption, total_scheduling_time


def main(args: Args):
    all_eval_configs = itertools.product(range(args.num_iterations), ALGORITHMS)
    results: list[dict[str, Any]] = []
    for seed_offset, (algorithm_name, algorithm) in tqdm(list(all_eval_configs)):
        scheduler = algorithm_strategy.get_scheduler(algorithm)
        makespan, energy_consumption, total_scheduling_time = run_algorithm(scheduler, seed_offset, args)
        results.append(
            {
                "Algorithm": algorithm_name,
                "Seed": seed_offset,
                "Makespan": makespan,
                "EnergyJ": energy_consumption,
                "Time": total_scheduling_time,
            }
        )

    df = DataFrame(results)
    ic(df)
    df.to_csv(args.export_csv)


if __name__ == "__main__":
    main(tyro.cli(Args))
