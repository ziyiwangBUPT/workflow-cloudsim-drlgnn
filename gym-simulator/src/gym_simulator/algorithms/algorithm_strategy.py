from gym_simulator.algorithms.base import BaseScheduler
from gym_simulator.algorithms.best_fit import BestFitScheduler
from gym_simulator.algorithms.cp_sat import CpSatScheduler
from gym_simulator.algorithms.fjssp import FjsspScheduler
from gym_simulator.algorithms.heft import HeftScheduler
from gym_simulator.algorithms.max_min import MaxMinScheduler
from gym_simulator.algorithms.min_min import MinMinScheduler
from gym_simulator.algorithms.round_robin import RoundRobinScheduler


def get_scheduler(algorithm: str) -> BaseScheduler:
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
    elif algorithm.startswith("fjssp_"):
        split_args = algorithm.split("_")
        assert len(split_args) == 3, "Invalid FJSSP algorithm format (expected: fjssp_<task_algo>_<vm_algo>)"
        _, task_select_algo, vm_select_algo = split_args
        return FjsspScheduler(task_select_algo=task_select_algo, vm_select_algo=vm_select_algo)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
