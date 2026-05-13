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

from scheduler.config.settings import ALGORITHMS, MIN_TESTING_DS_SEED
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
            EvaluationSetting(
                id=1,
                # dataset_args=DatasetArgs(
                #     host_count=3,
                #     vm_count=4,
                #     workflow_count=10,
                #     gnp_min_n=1,
                #     gnp_max_n=5,
                #     max_memory_gb=2,
                #     min_cpu_speed=2500,
                #     max_cpu_speed=5000,
                #     min_task_length=50_000,
                #     max_task_length=100_000,
                #     task_arrival="static",
                #     dag_method="gnp",
                # ),
                # dataset_args=DatasetArgs(
                #     host_count=3,
                #     vm_count=5,
                #     workflow_count=10,
                #     gnp_min_n=15,
                #     gnp_max_n=20,
                #     max_memory_gb=10,
                #     min_cpu_speed=1000,
                #     max_cpu_speed=5000,
                #     min_task_length=500,
                #     max_task_length=100_000,
                #     task_arrival="static",
                #     dag_method="gnp",
                # ),
                # dataset_args=DatasetArgs(
                #     host_count=3,
                #     vm_count=5,
                #     workflow_count=10,
                #     gnp_min_n=40,
                #     gnp_max_n=50,
                #     max_memory_gb=10,
                #     min_cpu_speed=1000,
                #     max_cpu_speed=5000,
                #     min_task_length=500,
                #     max_task_length=100_000,
                #     task_arrival="static",
                #     dag_method="gnp",
                # ),
                dataset_args=DatasetArgs(
                    host_count=4,
                    vm_count=5,
                    workflow_count=10,
                    gnp_min_n=10,
                    gnp_max_n=20,
                    max_memory_gb=10,
                    min_cpu_speed=1000,
                    max_cpu_speed=5000,
                    min_task_length=5000,
                    max_task_length=100_000,
                    task_arrival="static",
                    dag_method="gnp",
                ),
            ),
        ]
    )
    """number of tasks"""


@dataclass
class EvaluationSetting:
    id: int
    dataset_args: DatasetArgs

    def to_dataset_args(self):
        return self.dataset_args


def calculate_workflow_deadline_satisfaction_rate(solution: Solution) -> float:
    """
    计算工作流截止时间满足率

    对于每个工作流：
    1. 获取该工作流所有任务的完成时间（从vm_assignments中获取）
    2. 计算工作流的实际完成时间（最后一个任务的完成时间）
    3. 与预调度时计算的该工作流截止时间比较
    4. 计算满足率（实际完成时间 <= 截止时间的工作流数量 / 总工作流数量）

    Args:
        solution: 包含数据集和任务分配信息的解决方案

    Returns:
        工作流截止时间满足率（0.0 到 1.0 之间）
    """
    from scheduler.pre_scheduling.pre_computation import precompute_workflow_data
    from scheduler.pre_scheduling.ws_method import ContentionAwareWorkflowSequencing
    from scheduler.pre_scheduling.dp_method import BottleLayerAwareDeadlinePartition

    dataset = solution.dataset

    # 如果工作流还没有deadline，需要运行预调度来计算
    # 检查是否已有deadline
    need_prescheduling = any(wf.deadline == 0.0 for wf in dataset.workflows)

    if need_prescheduling:
        # 运行预调度计算工作流deadline
        ws_scheduler = ContentionAwareWorkflowSequencing(alpha1=0.33, alpha2=0.33, alpha3=0.33)
        dp_scheduler = BottleLayerAwareDeadlinePartition(beta=0.5)
        rho = 0.2  # 松弛因子

        # 阶段0：预计算每个工作流的平均完成时间和相关属性
        for workflow in dataset.workflows:
            precompute_workflow_data(workflow, dataset.vms, rho)

        # 阶段1：工作流优先级计算
        ws_scheduler.run(dataset.workflows, dataset.vms)

        # 阶段2：截止时间划分
        for workflow in dataset.workflows:
            dp_scheduler.run(workflow, dataset.vms)

    # 计算每个工作流的实际完成时间
    # 工作流ID -> 该工作流所有任务的完成时间列表
    workflow_completion_times: dict[int, list[float]] = {}

    for assignment in solution.vm_assignments:
        workflow_id = assignment.workflow_id
        end_time = assignment.end_time

        if workflow_id not in workflow_completion_times:
            workflow_completion_times[workflow_id] = []
        workflow_completion_times[workflow_id].append(end_time)

    # 计算每个工作流的实际完成时间（最后一个任务的完成时间）
    workflow_actual_completion_times: dict[int, float] = {}
    for workflow_id, completion_times in workflow_completion_times.items():
        if completion_times:
            workflow_actual_completion_times[workflow_id] = max(completion_times)

    # 计算满足率
    satisfied_count = 0
    total_count = 0

    for workflow in dataset.workflows:
        if workflow.deadline <= 0.0:
            # 如果deadline未设置或为0，跳过该工作流
            continue

        if workflow.id not in workflow_actual_completion_times:
            # 如果工作流没有任务分配，跳过
            continue

        total_count += 1

        actual_completion_time = workflow_actual_completion_times[workflow.id]
        # 工作流的实际完成时间应该从工作流开始时间（arrival_time）开始计算
        # deadline是相对于工作流开始的，所以需要计算相对完成时间
        workflow_start_time = workflow.arrival_time
        relative_completion_time = actual_completion_time - workflow_start_time

        # 检查是否满足截止时间（允许小的浮点误差）
        if relative_completion_time <= workflow.deadline + 1e-6:
            satisfied_count += 1

    # 计算满足率
    if total_count == 0:
        return 0.0

    satisfaction_rate = satisfied_count / total_count
    return satisfaction_rate


def run_algorithm(
        scheduler: BaseScheduler, seed_id: int, setting: EvaluationSetting, args: Args
) -> tuple[float, float, float, float]:
    env = CloudSimGymEnvironment(args.simulator, setting.to_dataset_args())

    total_scheduling_time: float = 0
    obs, info = env.reset(seed=MIN_TESTING_DS_SEED + seed_id)
    while True:
        scheduling_start_time = time.time()
        assignments = scheduler.schedule(obs.tasks, obs.vms)
        scheduling_end_time = time.time()
        total_scheduling_time += scheduling_end_time - scheduling_start_time
        obs, reward, terminated, truncated, info = env.step(SimEnvAction(assignments))
        if terminated or truncated:
            solution: Solution = info["solution"]

            # 计算总碳排放：遍历所有任务分配，累加每个任务的碳排放
            from scheduler.config.carbon_intensity import calculate_carbon_cost, FIXED_NUM_HOSTS
            from scheduler.rl_model.core.utils.helpers import active_energy_consumption_per_mi

            total_carbon_emission = 0.0

            # 创建task、vm和host的映射表以便快速查找
            task_map = {}
            for workflow in solution.dataset.workflows:
                for task in workflow.tasks:
                    task_map[(workflow.id, task.id)] = task

            vm_map = {vm.id: vm for vm in solution.dataset.vms}
            host_map = {host.id: host for host in solution.dataset.hosts}

            # 遍历所有任务分配，计算每个任务的碳排放
            for assignment in solution.vm_assignments:
                # 查找任务
                task_key = (assignment.workflow_id, assignment.task_id)
                if task_key not in task_map:
                    continue
                task = task_map[task_key]

                # 查找VM
                if assignment.vm_id not in vm_map:
                    continue
                vm = vm_map[assignment.vm_id]

                # 查找对应的Host
                if vm.host_id not in host_map:
                    continue
                host = host_map[vm.host_id]

                # 计算任务能耗（Joules）
                # 公式: 能耗 = (host峰值功率 - host空闲功率) / host_CPU速度 * 任务长度
                # 这与 active_energy_consumption_per_mi 的逻辑一致：
                # per_mi_energy = (power_peak - power_idle) / host_cpu_speed
                # total_energy = per_mi_energy * task_length
                if host.cpu_speed_mips > 0:
                    energy_joules = (host.power_peak_watt - host.power_idle_watt) / host.cpu_speed_mips * task.length
                else:
                    energy_joules = 0

                # 确保host_id在有效范围内
                safe_host_id = vm.host_id % FIXED_NUM_HOSTS

                # 计算该任务的碳排放（gCO2）
                task_carbon_cost = calculate_carbon_cost(
                    energy_joules=energy_joules,
                    host_id=safe_host_id,
                    start_time=assignment.start_time,
                    end_time=assignment.end_time
                )

                # 累加到总碳排放
                total_carbon_emission += task_carbon_cost

            # 计算makespan
            if solution.vm_assignments:
                start_time = min([assignment.start_time for assignment in solution.vm_assignments])
                end_time = max([assignment.end_time for assignment in solution.vm_assignments])
                makespan = end_time - start_time
            else:
                makespan = 0.0

            # 计算工作流截止时间满足率
            workflow_deadline_satisfaction_rate = calculate_workflow_deadline_satisfaction_rate(
                solution
            )

            return makespan, total_carbon_emission, total_scheduling_time, workflow_deadline_satisfaction_rate


def main(args: Args):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    all_eval_configs = itertools.product(range(args.num_samples_per_setting), args.settings, ALGORITHMS)
    results: list[dict[str, Any]] = []
    for seed_id, setting, (algorithm_name, algorithm) in tqdm(list(all_eval_configs)):
        scheduler = algorithm_strategy.get_scheduler(algorithm)
        makespan, carbon_emission, total_scheduling_time, workflow_deadline_satisfaction_rate = run_algorithm(
            scheduler, seed_id, setting, args
        )
        results.append(
            {
                "SeedId": seed_id,
                "SettingId": setting.id,
                "Algorithm": algorithm_name,
                "Makespan": makespan,
                "CarbonEmission_gCO2": carbon_emission,  # 总碳排放（gCO2）
                "Time": total_scheduling_time,
                "WorkflowDeadlineSatisfactionRate": workflow_deadline_satisfaction_rate,  # 工作流截止时间满足率（0.0-1.0）
            }
        )

    df = DataFrame(results)
    ic(df)
    df.to_csv(args.export_csv)


if __name__ == "__main__":
    main(tyro.cli(Args))
