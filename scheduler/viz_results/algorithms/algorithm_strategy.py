from scheduler.viz_results.algorithms.base import BaseScheduler
from scheduler.viz_results.algorithms.best_fit import BestFitScheduler
from scheduler.viz_results.algorithms.cp_sat import CpSatScheduler
from scheduler.viz_results.algorithms.heft import HeftScheduler
from scheduler.viz_results.algorithms.max_min import MaxMinScheduler
from scheduler.viz_results.algorithms.min_min import MinMinScheduler
from scheduler.viz_results.algorithms.power_saving import PowerSavingScheduler
from scheduler.viz_results.algorithms.round_robin import RoundRobinScheduler


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
    elif algorithm == "power_saving":
        return PowerSavingScheduler()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
