import itertools
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
from gym_simulator.args import EVALUATING_DS1_ARGS
from gym_simulator.core.simulators.proxy import InternalProxySimulatorObs
from gym_simulator.environments.static import StaticCloudSimEnvironment


@dataclasses.dataclass
class Args:
    simulator: str
    """path to the simulator JAR file"""
    seed: int
    """random seed"""
    buffer_size: int
    """size of the workflow scheduler buffer"""
    buffer_timeout: int
    """Timeout of the workflow scheduler buffer"""
    file: str
    """File to output the export CSV"""
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
    ("Makespan Heuristic", "rlh:makespan"),
    ("Energy Heuristic", "rlh:energy"),
    # TODO: ACO
    ("Model A", "rl:gin:1732021759_ppo_gin_makespan_power_est_10_20:model_1064960.pt"),
    ("Model B", "rl:gin:1732268581_ppo_exp_1:model_163840.pt"),
]


# Main
# -----------------------------------------------------------------------------


def main(args: Args):
    env_config = {
        "simulator_mode": "embedded",
        "simulator_kwargs": {
            "dataset_args": dataclasses.asdict(EVALUATING_DS1_ARGS),
            "simulator_jar_path": args.simulator,
            "scheduler_preset": f"buffer:gym:{args.buffer_size}:{args.buffer_timeout}",
            "verbose": False,
            "remote_debug": False,
        },
    }
    agent_env_config = {
        "simulator_mode": "proxy",
        "simulator_kwargs": {"proxy_obs": InternalProxySimulatorObs()},
    }

    all_eval_configs = itertools.product(range(args.num_iterations), ALGORITHMS)
    results: list[dict[str, Any]] = []
    for seed_offset, (algorithm_name, algorithm) in tqdm(list(all_eval_configs)):
        current_seed = args.seed + seed_offset
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        torch.backends.cudnn.deterministic = True
        env_config["seed"] = current_seed
        agent_env_config["seed"] = current_seed

        scheduler = algorithm_strategy.get_scheduler(algorithm, env_config=copy.deepcopy(agent_env_config))
        total_scheduling_time = 0

        # Run environment
        env = StaticCloudSimEnvironment(env_config=copy.deepcopy(env_config))
        (tasks, vms), _ = env.reset(seed=current_seed)
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
                "Algorithm": algorithm_name,
                "Seed": current_seed,
                "Makespan": makespan,
                "PowerW": power_watt,
                "EnergyJ": power_watt * makespan,
                "Time": total_scheduling_time,
            }
        )

    df = DataFrame(results)
    icecream.ic(df)
    df.to_csv(args.file)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
