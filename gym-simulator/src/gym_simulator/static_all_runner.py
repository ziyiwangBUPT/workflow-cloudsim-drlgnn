from typing import Any
import tyro
import copy
import dataclasses
import random
import time

from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np

from dataset_generator.core.models import Solution
from dataset_generator.visualizers.plotters import plot_gantt_chart
from gym_simulator.algorithms import algorithm_strategy
from gym_simulator.environments.static import StaticCloudSimEnvironment


@dataclasses.dataclass
class Args:
    simulator: str
    """path to the simulator JAR file"""
    seed: int = 0
    """random seed"""
    host_count: int = 10
    """number of hosts"""
    vm_count: int = 10
    """number of VMs"""
    workflow_count: int = 5
    """number of workflows"""
    task_limit: int = 5
    """maximum number of tasks"""
    gantt_chart_prefix: str = "tmp/gantt_chart"
    """prefix for the Gantt chart files"""


def main(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    env_config = {
        "host_count": args.host_count,
        "vm_count": args.vm_count,
        "workflow_count": args.workflow_count,
        "task_limit": args.task_limit,
        "simulator_mode": "embedded",
        "seed": args.seed,
        "simulator_kwargs": {
            "simulator_jar_path": args.simulator,
            "verbose": False,
            "remote_debug": False,
        },
    }
    algorithms = [
        "round_robin",
        "max_min",
        "min_min",
        "best_fit",
        "fjssp_fifo_spt",
        "fjssp_fifo_eet",
        "fjssp_mopnr_spt",
        "fjssp_mopnr_eet",
        "fjssp_lwkr_spt",
        "fjssp_lwkr_eet",
        "fjssp_mwkr_spt",
        "fjssp_mwkr_eet",
        "cp_sat",
        "heft",
        "heft_one",
        "power_saving",
        "rl_static",
        "rl_vmonly_heftonereward_1728550737",  # RL - trained on heft_one reward: reward = 0, final reward = (baseline - makespan) / baseline
        "rl_vmonly_makespandiffreward_1728678033",  # RL - trained on makespan diff: reward = oldmakespan - makespan, final reward = -makespan
        "rl_vmonly_roundrobinreward_1728758255",  # RL - trained on round_robin reward: reward = 0, final reward = -makespan / baseline
        "rl_rl_vm_clean_rl_train__1__1728834561",
    ]

    stats: list[dict[str, Any]] = []
    for algorithm in algorithms:
        env = StaticCloudSimEnvironment(env_config=copy.deepcopy(env_config))
        scheduler = algorithm_strategy.get_scheduler(algorithm, env_config=copy.deepcopy(env_config))

        (tasks, vms), _ = env.reset(seed=args.seed)
        t1 = time.time()
        action = scheduler.schedule(tasks, vms)
        t2 = time.time()
        _, reward, terminated, truncated, info = env.step(action)
        assert terminated or truncated, "Static environment should terminate after one step"

        solution = info.get("solution")
        power_watt = info.get("total_power_consumption_watt")
        assert solution is not None and isinstance(solution, Solution), "Solution is not available"
        fig, ax = plt.subplots()
        plot_gantt_chart(ax, solution.dataset.workflows, solution.dataset.vms, solution.vm_assignments, label=True)
        fig.set_figwidth(12)
        fig.set_figheight(7)
        fig.tight_layout()
        plt.savefig(f"{args.gantt_chart_prefix}_{algorithm}.png")
        plt.close(fig)

        makespan = max([assignment.end_time for assignment in solution.vm_assignments])
        entry = {
            "Algorithm": algorithm,
            "Reward": reward,
            "Makespan": makespan,
            "Time": t2 - t1,
            "IsOptimal": scheduler.is_optimal(),
            "PowerW": power_watt,
        }
        print(entry)
        stats.append(entry)

    env.close()

    # Plotting the comparison
    df = DataFrame(stats).sort_values(by="Makespan", ascending=True).reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(14, 7))
    bar_width = 0.25
    index = range(len(df["Algorithm"]))

    # Plotting Makespan
    ax1.bar(index, df["Makespan"], width=bar_width, label="Makespan", color="tab:blue")
    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel("Makespan", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks([i + bar_width / 2 for i in index])
    # Algorithm name should be df["Algorithm"] for non-optimal solutions, and df["Algorithm*"] for optimal solutions
    algorithm_names = [f"{df['Algorithm'][i]}*" if df["IsOptimal"][i] else df["Algorithm"][i] for i in index]
    ax1.set_xticklabels(algorithm_names, rotation=45, ha="right")

    # Adding Energy consumption to the plot
    ax2 = ax1.twinx()
    ax2.bar([i + bar_width for i in index], df["PowerW"], width=bar_width, label="Power (W)", color="tab:green")
    ax2.set_ylabel("Power (W)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    # Creating a secondary y-axis for Time (log scale)
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.2))
    ax3.bar([i + 2 * bar_width for i in index], df["Time"], width=bar_width, label="Time (s)", color="tab:red")
    ax3.set_ylabel("Time (s)", color="tab:red")
    ax3.set_yscale("log")  # Set log scale for time
    ax3.tick_params(axis="y", labelcolor="tab:red")

    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.title("Comparison of Algorithms by Makespan, Power and Time")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
