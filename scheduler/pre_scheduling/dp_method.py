"""
截止时间划分算法 (Deadline Partition)
改编自 ecmws-experiments/methods/dp_method.py
"""
from typing import List, Dict
from scheduler.dataset_generator.core.models import Task, Workflow, Vm
from scheduler.pre_scheduling.methods_proto import DeadlinePartition
from scheduler.pre_scheduling.pre_computation import topological_sort, estimate_task_avg_work_time


def compute_task_level(tasks: List[Task], task_map: Dict[int, Task]) -> Dict[int, int]:
    """
    计算每个任务的层级 (level)
    层级定义：从任务到出口任务（没有后继的任务）的最长路径长度
    
    算法：从后向前遍历任务，递归计算
    - 如果任务没有后继，层级为 1
    - 否则，层级 = max(所有后继的层级) + 1
    
    Args:
        tasks: 任务列表（应该是拓扑排序后的）
        task_map: task_id -> Task对象的映射
        
    Returns:
        task_id -> level 的映射
    """
    levels = {}
    
    # 从后向前遍历（逆拓扑序）
    for task in reversed(tasks):
        # 如果没有后继任务，层级为1
        if not task.child_ids:
            levels[task.id] = 1
        else:
            # 层级 = max(所有后继任务的层级) + 1
            child_levels = []
            for child_id in task.child_ids:
                if child_id in task_map and child_id in levels:
                    child_levels.append(levels[child_id])
            
            if child_levels:
                levels[task.id] = max(child_levels) + 1
            else:
                levels[task.id] = 1
    
    return levels


def compute_task_level_phi(level_map: Dict[int, int]) -> Dict[int, int]:
    """
    计算每个层级的任务数量
    
    Args:
        level_map: task_id -> level 的映射
        
    Returns:
        level -> 该层级的任务数量的映射
    """
    levels_phi = {}
    for task_id, level in level_map.items():
        levels_phi[level] = levels_phi.get(level, 0) + 1
    return levels_phi


def compute_rank_dp(tasks: List[Task], 
                    task_map: Dict[int, Task],
                    beta: float,
                    levels: Dict[int, int],
                    levels_phi: Dict[int, int],
                    task_avg_worktime: Dict[int, float],
                    use_beta: bool = False) -> Dict[int, float]:
    """
    计算每个任务的 DP 排名 (rank_dp)
    
    rank_dp 是一个优先级分数，用于确定任务的调度顺序
    计算公式（从后向前递归）：
    - 如果任务没有后继：rank_dp = avg_worktime
    - 否则：rank_dp = max(rank_dp[succ] + transtime[succ]) + avg_worktime
    
    Args:
        tasks: 任务列表（应该是拓扑排序后的）
        task_map: task_id -> Task对象的映射
        beta: 瓶颈层调整因子
        levels: task_id -> level 的映射
        levels_phi: level -> 任务数量的映射
        task_avg_worktime: task_id -> 平均工作时间的映射
        use_beta: 是否使用 beta 调整（默认不使用）
        
    Returns:
        task_id -> rank_dp 的映射
    """
    rank_dp = {}
    
    # 从后向前遍历（逆拓扑序）
    for task in reversed(tasks):
        if not task.child_ids:
            # 出口任务：rank_dp = avg_worktime
            rank_dp[task.id] = task_avg_worktime[task.id]
        else:
            level = levels[task.id]
            
            if level == 1:
                # 最后一层的任务
                rank_dp[task.id] = task_avg_worktime[task.id]
            else:
                # 计算所有后继任务的最大 rank_dp
                # 注意：paper1115 中没有传输时间，所以这里 transtime = 0
                max_rank_dp = 0
                for child_id in task.child_ids:
                    if child_id in task_map and child_id in rank_dp:
                        # transtime = 0（paper1115中没有数据传输）
                        max_rank_dp = max(max_rank_dp, rank_dp[child_id])
                
                if use_beta:
                    # 使用瓶颈层调整
                    phi_ratio = levels_phi[level] / levels_phi[level - 1]
                    rank_dp[task.id] = max_rank_dp + task_avg_worktime[task.id] + beta ** phi_ratio
                else:
                    rank_dp[task.id] = max_rank_dp + task_avg_worktime[task.id]
    
    return rank_dp


def compute_sub_deadlines(tasks: List[Task],
                          task_map: Dict[int, Task],
                          beta: float,
                          workflow: Workflow,
                          task_avg_worktime: Dict[int, float]) -> None:
    """
    计算并设置每个任务的子截止时间 (sub-deadline)
    这是 DP 算法的核心功能
    
    算法步骤：
    1. 计算任务层级
    2. 计算每层的任务数量
    3. 计算每个任务的 rank_dp
    4. 根据 rank_dp 分配子截止时间
    
    子截止时间公式：
    task.deadline = workflow.deadline * (rank_dp_0 - rank_dp[task] + avg_worktime[task]) / rank_dp_0
    
    其中 rank_dp_0 是入口任务（没有前驱的任务）的最大 rank_dp
    
    Args:
        tasks: 任务列表（应该是拓扑排序后的）
        task_map: task_id -> Task对象的映射
        beta: 瓶颈层调整因子
        workflow: 工作流对象
        task_avg_worktime: task_id -> 平均工作时间的映射
    """
    # 1. 计算任务层级
    level_map = compute_task_level(tasks, task_map)
    
    # 2. 计算每层的任务数量
    level_phi = compute_task_level_phi(level_map)
    
    # 3. 计算 rank_dp
    rank_dp_map = compute_rank_dp(tasks, task_map, beta, level_map, level_phi, 
                                   task_avg_worktime, use_beta=False)
    
    # 4. 找出入口任务的最大 rank_dp
    rank_dp_0 = max(
        rank_dp_map[task.id] for task in tasks if not task.parent_ids
    )
    
    # 避免除零错误
    if rank_dp_0 <= 0:
        rank_dp_0 = 1.0
    
    # 5. 为每个任务设置 rank_dp 和 deadline
    for task in tasks:
        # 设置优先级分数
        task.rank_dp = rank_dp_map[task.id]
        
        # 计算子截止时间
        task.deadline = workflow.deadline * (
            rank_dp_0 - rank_dp_map[task.id] + task_avg_worktime[task.id]
        ) / rank_dp_0
        
        # 计算松弛时间（可选）
        task.slack_time = task.deadline - task.avg_eft


class BottleLayerAwareDeadlinePartition(DeadlinePartition):
    """
    基于瓶颈层的截止时间划分算法
    
    该算法为工作流中的每个任务分配一个子截止时间，考虑了：
    1. 任务的关键路径长度 (rank_dp)
    2. 任务的层级结构
    3. 任务的平均执行时间
    """
    
    def __init__(self, beta: float):
        """
        初始化截止时间划分算法
        
        Args:
            beta: 瓶颈层调整因子（本实现中未使用，但保留接口）
        """
        self.beta = beta
    
    def run(self, workflow: Workflow, vms: List[Vm]) -> None:
        """
        运行截止时间划分算法
        该方法会为工作流中的每个任务设置 rank_dp 和 deadline 属性
        
        Args:
            workflow: 工作流对象
            vms: 虚拟机列表
        """
        # 1. 拓扑排序
        sorted_tasks = topological_sort(workflow)
        
        # 2. 建立任务映射
        task_map = {task.id: task for task in workflow.tasks}
        
        # 3. 计算平均工作时间
        task_avg_worktime = estimate_task_avg_work_time(workflow, vms)
        
        # 4. 计算子截止时间
        compute_sub_deadlines(sorted_tasks, task_map, self.beta, workflow, task_avg_worktime)

