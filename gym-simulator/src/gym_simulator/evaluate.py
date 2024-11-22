import time
from typing import Any
import icecream
import torch
import tyro
import copy
import dataclasses
import random

from tqdm import tqdm
from pandas import DataFrame
import numpy as np

from gym_simulator.algorithms import algorithm_strategy
from gym_simulator.core.simulators.proxy import InternalProxySimulatorObs
from gym_simulator.environments.static import StaticCloudSimEnvironment


@dataclasses.dataclass
class Args:
    simulator: str
    """path to the simulator JAR file"""
    seed: int
    """random seed"""
    host_count: int
    """number of hosts"""
    vm_count: int
    """number of VMs"""
    workflow_count: int
    """number of workflows"""
    task_limit: int
    """maximum number of tasks"""
    buffer_size: int
    """size of the workflow scheduler buffer"""
    buffer_timeout: int
    """Timeout of the workflow scheduler buffer"""
    num_iterations: int = 100
    """Number of iterations to evaluate"""


ALGORITHMS = [
    ("CP-SAT", "cp_sat"),
    ("Round Robin", "round_robin"),
    ("Min-Min", "min_min"),
    ("Best Fit", "best_fit"),
    ("Max-Min", "max_min"),
    ("HEFT", "heft"),
    ("Power Heuristic", "power_saving"),
    # TODO: Makespan heuristic
    # TODO: ACO
    ("Proposed Model", "rl:gin:1732021759_ppo_gin_makespan_power_est_10_20:model_1064960.pt"),
]


# Run Test
# -----------------------------------------------------------------------------


def run_test(test_id: int, env_config: dict[str, Any], agent_env_config: dict[str, Any]):
    random.seed(test_id)
    np.random.seed(test_id)
    torch.manual_seed(test_id)
    torch.backends.cudnn.deterministic = True

    env_config["seed"] = test_id
    agent_env_config["seed"] = test_id

    results: list[dict[str, Any]] = []
    for name, algorithm in ALGORITHMS:
        scheduler = algorithm_strategy.get_scheduler(algorithm, env_config=copy.deepcopy(agent_env_config))
        total_scheduling_time = 0

        # Run environment
        env = StaticCloudSimEnvironment(env_config=copy.deepcopy(env_config))
        (tasks, vms), _ = env.reset(seed=test_id)
        while True:
            scheduling_start_time = time.time()
            action = scheduler.schedule(tasks, vms)
            scheduling_end_time = time.time()
            total_scheduling_time = scheduling_end_time - scheduling_start_time
            (tasks, vms), _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()

        # Append result
        solution = info.get("solution")
        power_watt = info.get("total_power_consumption_watt")
        makespan = max([assignment.end_time for assignment in solution.vm_assignments])
        results.append(
            {
                "Algorithm": name,
                "Seed": test_id,
                "Makespan": makespan,
                "PowerW": power_watt,
                "EnergyJ": power_watt * makespan,
                "Time": total_scheduling_time,
            }
        )
    return results


# Main
# -----------------------------------------------------------------------------


def main(args: Args):
    env_config = {
        "host_count": args.host_count,
        "vm_count": args.vm_count,
        "workflow_count": args.workflow_count,
        "task_limit": args.task_limit,
        "simulator_mode": "embedded",
        "simulator_kwargs": {
            "dataset_args": {
                "task_arrival": "dynamic",
            },
            "simulator_jar_path": args.simulator,
            "scheduler_preset": f"buffer:gym:{args.buffer_size}:{args.buffer_timeout}",
            "verbose": False,
            "remote_debug": False,
        },
    }
    agent_env_config = {
        "host_count": args.host_count,
        "vm_count": args.vm_count,
        "workflow_count": args.workflow_count,
        "task_limit": args.task_limit,
        "simulator_mode": "proxy",
        "simulator_kwargs": {"proxy_obs": InternalProxySimulatorObs()},
    }

    results = []
    for seed_offset in tqdm(range(args.num_iterations)):
        current_seed = args.seed + seed_offset
        results.extend(run_test(current_seed, env_config, agent_env_config))

    df = DataFrame(results)
    icecream.ic(df)
    df.to_csv("logs/data/test.csv")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
