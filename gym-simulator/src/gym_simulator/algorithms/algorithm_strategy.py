from typing import Any
from gym_simulator.algorithms.base import BaseScheduler
from gym_simulator.algorithms.best_fit import BestFitScheduler
from gym_simulator.algorithms.cp_sat import CpSatScheduler
from gym_simulator.algorithms.fjssp import FjsspScheduler
from gym_simulator.algorithms.heft import HeftScheduler
from gym_simulator.algorithms.heft_one import HeftOneScheduler
from gym_simulator.algorithms.max_min import MaxMinScheduler
from gym_simulator.algorithms.min_min import MinMinScheduler
from gym_simulator.algorithms.power_saving import PowerSavingScheduler
from gym_simulator.algorithms.rl_static import RlStaticScheduler
from gym_simulator.algorithms.round_robin import RoundRobinScheduler


def get_scheduler(algorithm: str, env_config: dict[str, Any] | None = None) -> BaseScheduler:
    if algorithm == "round_robin":
        return RoundRobinScheduler()
    elif algorithm == "best_fit":
        return BestFitScheduler()
    elif algorithm == "min_min":
        return MinMinScheduler()
    elif algorithm == "max_min":
        return MaxMinScheduler()
    elif algorithm == "cp_sat":
        return CpSatScheduler()
    elif algorithm == "heft":
        return HeftScheduler()
    elif algorithm == "heft_one":
        return HeftOneScheduler()
    elif algorithm == "power_saving":
        return PowerSavingScheduler()
    elif algorithm == "rl_static":
        assert env_config is not None, "env_config is required for RL algorithm"
        return RlStaticScheduler(env_config)
    elif algorithm.startswith("fjssp_"):
        split_args = algorithm.split("_")
        assert len(split_args) == 3, "Invalid FJSSP algorithm format (expected: fjssp_<task_algo>_<vm_algo>)"
        _, task_select_algo, vm_select_algo = split_args
        return FjsspScheduler(task_select_algo=task_select_algo, vm_select_algo=vm_select_algo)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
