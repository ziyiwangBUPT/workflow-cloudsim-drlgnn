import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import tyro
import numpy as np
from icecream import ic
from pandas import DataFrame
from tqdm import tqdm

from scheduler.config.settings import ALGORITHMS, MIN_EVALUATING_DS_SEED
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
    export_csv: str
    """file to output the export CSV"""
    num_samples_per_setting: int = 10
    """number of iterations to evaluate in a setting"""
    settings: list["EvaluationSetting"] = field(
        default_factory=lambda: [
            EvaluationSetting(id=0, num_tasks=200),
            EvaluationSetting(id=1, num_tasks=300),
            EvaluationSetting(id=2, num_tasks=400),
        ]
    )
    """number of tasks"""


@dataclass
class EvaluationSetting:
    id: int
    num_tasks: int


def create_dataset(num_tasks: int) -> DatasetArgs:
    return DatasetArgs(
        host_count=10,
        vm_count=4,
        workflow_count=1,
        gnp_min_n=num_tasks,
        gnp_max_n=num_tasks,
        max_memory_gb=10,
        min_cpu_speed=500,
        max_cpu_speed=5000,
        min_task_length=500,
        max_task_length=100_000,
        task_arrival="static",
        dag_method="gnp",
    )


def run_algorithm(
    scheduler: BaseScheduler, seed_id: int, setting: EvaluationSetting, args: Args
) -> tuple[float, float, float]:
    env = CloudSimGymEnvironment(args.simulator, create_dataset(setting.num_tasks))

    total_scheduling_time: float = 0
    obs, info = env.reset(seed=MIN_EVALUATING_DS_SEED + seed_id)
    while True:
        scheduling_start_time = time.time()
        assignments = scheduler.schedule(obs.tasks, obs.vms)
        scheduling_end_time = time.time()
        total_scheduling_time += scheduling_end_time - scheduling_start_time
        obs, reward, terminated, truncated, info = env.step(SimEnvAction(assignments))
        if terminated or truncated:
            solution: Solution = info["solution"]
            energy_consumption: float = info.get("active_energy_consumption_j", 0)
            start_time = min([assignment.start_time for assignment in solution.vm_assignments])
            end_time = max([assignment.end_time for assignment in solution.vm_assignments])
            makespan = end_time - start_time
            return makespan, energy_consumption, total_scheduling_time


def main(args: Args):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    all_eval_configs = itertools.product(range(args.num_samples_per_setting), args.settings, ALGORITHMS)
    results: list[dict[str, Any]] = []
    for seed_id, setting, (algorithm_name, algorithm) in tqdm(list(all_eval_configs)):
        scheduler = algorithm_strategy.get_scheduler(algorithm)
        makespan, energy_consumption, total_scheduling_time = run_algorithm(scheduler, seed_id, setting, args)
        results.append(
            {
                "Algorithm": algorithm_name,
                "Index": setting.id,
                "SeedId": seed_id,
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
