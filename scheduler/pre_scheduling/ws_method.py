"""
工作流排序算法 (Workflow Sequencing)
改编自 ecmws-experiments/methods/ws_method.py
"""
from typing import List
from scheduler.dataset_generator.core.models import Workflow, Vm
from scheduler.pre_scheduling.methods_proto import WorkflowSequencing


def compute_contention(workflow: Workflow) -> int:
    """
    计算工作流的竞争度 (contention)
    竞争度是指在工作流执行期间，同时活跃的最大任务数
    
    算法思路：
    1. 将所有任务按照 avg_est 排序，形成时间区间列表
    2. 对于每个任务，计算与它的执行时间区间 [avg_est, avg_eft] 重叠的任务数量
    3. 返回最大重叠数量
    
    Args:
        workflow: 工作流对象
        
    Returns:
        竞争度（最大并发任务数）
    """
    # 创建任务的时间区间列表 (task, avg_est, avg_eft)
    intervals = [(task, task.avg_est, task.avg_eft) for task in workflow.tasks]
    # 按 avg_est 排序
    intervals.sort(key=lambda x: x[1])
    
    contention = 0
    
    while intervals:
        cur_contention = 1
        task, est, eft = intervals.pop(0)
        
        overlapping_tasks = []
        for idx, (other_task, other_est, other_eft) in enumerate(intervals):
            # 如果当前任务的结束时间 >= 其他任务的开始时间，说明有重叠
            if eft >= other_est:
                cur_contention += 1
                overlapping_tasks.append(idx)
            else:
                # 因为已经排序，后面的任务不会再重叠
                break
        
        # 移除已经计算过的重叠任务
        for idx in sorted(overlapping_tasks, reverse=True):
            del intervals[idx]
        
        contention = max(contention, cur_contention)
    
    return contention


class ContentionAwareWorkflowSequencing(WorkflowSequencing):
    """
    基于竞争度的工作流排序算法
    
    该算法综合考虑三个因素对工作流进行排序：
    1. 松弛时间 (slack time): st = deadline - avg_eft
    2. 工作负载 (workload): wl = sum of all task lengths
    3. 竞争度 (contention): ct = 最大并发任务数
    
    排序公式: rank = a1 * (st/max_st) + a2 * (wl/max_wl) + a3 * (ct/max_ct)
    rank 越小的工作流优先级越高（越先执行）
    """
    
    def __init__(self, alpha1: float, alpha2: float, alpha3: float):
        """
        初始化工作流排序算法
        
        Args:
            alpha1: 松弛时间的权重
            alpha2: 工作负载的权重
            alpha3: 竞争度的权重
        """
        self.a1 = alpha1
        self.a2 = alpha2
        self.a3 = alpha3
    
    def run(self, workflows: List[Workflow], vms: List[Vm]) -> List[Workflow]:
        """
        运行工作流排序算法
        
        Args:
            workflows: 工作流列表
            vms: 虚拟机列表（在此算法中未直接使用，但保留接口一致性）
            
        Returns:
            排序后的工作流列表
        """
        if not workflows:
            return []
        
        # 1. 计算每个工作流的竞争度
        ct_lst = [compute_contention(wf) for wf in workflows]
        
        # 2. 获取每个工作流的松弛时间和工作负载
        st_lst = [wf.avg_slacktime for wf in workflows]
        wl_lst = [wf.workload for wf in workflows]
        
        # 3. 计算归一化因子（最大值）
        max_ct = max(ct_lst) if max(ct_lst) > 0 else 1
        max_st = max(st_lst) if max(st_lst) > 0 else 1
        max_wl = max(wl_lst) if max(wl_lst) > 0 else 1
        
        # 4. 计算每个工作流的排序分数
        ranks = [
            (wf, self.a1 * st / max_st + self.a2 * wl / max_wl + self.a3 * ct / max_ct)
            for st, wl, ct, wf in zip(st_lst, wl_lst, ct_lst, workflows)
        ]
        
        # 5. 按分数升序排序（分数越小优先级越高）
        ranks.sort(key=lambda x: x[1])
        
        # 6. 返回排序后的工作流列表
        return [wf for wf, _ in ranks]


class RandomWorkflowSequencing(WorkflowSequencing):
    """
    随机工作流排序算法（用于对比实验）
    """
    
    def run(self, workflows: List[Workflow], vms: List[Vm]) -> List[Workflow]:
        """
        随机打乱工作流顺序
        
        Args:
            workflows: 工作流列表
            vms: 虚拟机列表（未使用）
            
        Returns:
            随机排序后的工作流列表
        """
        import random
        workflows_copy = [wf for wf in workflows]
        random.shuffle(workflows_copy)
        return workflows_copy

