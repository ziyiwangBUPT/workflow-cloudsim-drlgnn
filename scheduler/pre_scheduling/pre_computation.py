"""
预计算模块
用于估算任务和工作流的平均执行时间
改编自 ecmws-experiments/methods/time_estimation.py
"""
from typing import List, Dict
from scheduler.dataset_generator.core.models import Task, Workflow, Vm


def compute_avg_speed(vms: List[Vm]) -> float:
    """
    计算所有虚拟机的平均计算速度
    
    Args:
        vms: 虚拟机列表
        
    Returns:
        平均速度 (MIPS)
    """
    if not vms:
        return 1.0
    
    total_speed = sum(vm.cpu_speed_mips for vm in vms)
    return total_speed / len(vms)


def build_task_relationships(workflow: Workflow) -> Dict[int, List[Task]]:
    """
    构建任务的父子关系映射
    
    Args:
        workflow: 工作流对象
        
    Returns:
        task_id -> Task对象的映射
    """
    # 建立 task_id 到 Task 对象的映射
    task_map = {task.id: task for task in workflow.tasks}
    
    # 为每个任务设置 parent_ids
    for task in workflow.tasks:
        task.parent_ids = []
    
    # 遍历所有任务，根据 child_ids 反推 parent_ids
    for task in workflow.tasks:
        for child_id in task.child_ids:
            if child_id in task_map:
                task_map[child_id].parent_ids.append(task.id)
    
    return task_map


def topological_sort(workflow: Workflow) -> List[Task]:
    """
    对工作流中的任务进行拓扑排序
    
    Args:
        workflow: 工作流对象
        
    Returns:
        拓扑排序后的任务列表
    """
    task_map = build_task_relationships(workflow)
    
    # 计算每个任务的入度
    in_degree = {task.id: len(task.parent_ids) for task in workflow.tasks}
    
    # 找出所有入度为0的任务（起始任务）
    queue = [task for task in workflow.tasks if in_degree[task.id] == 0]
    sorted_tasks = []
    
    while queue:
        # 取出一个入度为0的任务
        current = queue.pop(0)
        sorted_tasks.append(current)
        
        # 遍历当前任务的所有子任务
        for child_id in current.child_ids:
            if child_id in task_map:
                child_task = task_map[child_id]
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_task)
    
    return sorted_tasks


def estimate_task_avg_work_time(workflow: Workflow, vms: List[Vm]) -> Dict[int, float]:
    """
    估算每个任务在平均资源上的工作时间
    
    Args:
        workflow: 工作流对象
        vms: 虚拟机列表
        
    Returns:
        task_id -> 平均工作时间的映射
    """
    avg_speed = compute_avg_speed(vms)
    task_avg_worktime = {}
    
    for task in workflow.tasks:
        # 使用 length / avg_speed 计算平均执行时间
        task_avg_worktime[task.id] = task.length / avg_speed if avg_speed > 0 else task.length
    
    return task_avg_worktime


def estimate_task_avg_eft(task: Task, task_map: Dict[int, Task], 
                          task_avg_worktime: Dict[int, float], 
                          workflow: Workflow) -> None:
    """
    估算任务的平均最早完成时间 (Average Earliest Finish Time)
    
    Args:
        task: 当前任务
        task_map: task_id -> Task对象的映射
        task_avg_worktime: task_id -> 平均工作时间的映射
        workflow: 工作流对象
    """
    # 计算 avg_est (平均最早开始时间)
    # avg_est 是所有前驱任务的 avg_eft 的最大值
    if task.parent_ids:
        task.avg_est = max(task_map[parent_id].avg_eft for parent_id in task.parent_ids)
    else:
        # 如果没有前驱任务，则从工作流到达时间开始
        task.avg_est = workflow.arrival_time
    
    # avg_eft = avg_est + 平均工作时间
    # 注意：paper1115 中没有传输时间的概念，所以这里不考虑传输延迟
    task.avg_eft = task.avg_est + task_avg_worktime[task.id]


def estimate_workflow_avg_eft(workflow: Workflow, vms: List[Vm]) -> None:
    """
    估算工作流的平均最早完成时间，并填充所有任务的 avg_est 和 avg_eft
    
    Args:
        workflow: 工作流对象
        vms: 虚拟机列表
    """
    # 1. 建立任务关系映射
    task_map = build_task_relationships(workflow)
    
    # 2. 拓扑排序
    sorted_tasks = topological_sort(workflow)
    
    # 3. 计算每个任务的平均工作时间
    task_avg_worktime = estimate_task_avg_work_time(workflow, vms)
    
    # 4. 按拓扑顺序计算每个任务的 avg_est 和 avg_eft
    avg_eft = 0
    for task in sorted_tasks:
        estimate_task_avg_eft(task, task_map, task_avg_worktime, workflow)
        avg_eft = max(avg_eft, task.avg_eft)
    
    # 5. 设置工作流的 avg_eft
    workflow.avg_eft = avg_eft


def precompute_workflow_data(workflow: Workflow, vms: List[Vm], rho: float = 0.2) -> None:
    """
    预计算工作流的所有必要数据
    这是在预调度阶段必须调用的函数
    
    Args:
        workflow: 工作流对象
        vms: 虚拟机列表
        rho: 松弛因子，用于计算 deadline = avg_eft * (1 + rho)
    """
    # 1. 估算工作流的平均完成时间（同时也会计算所有任务的 avg_est 和 avg_eft）
    estimate_workflow_avg_eft(workflow, vms)
    
    # 2. 计算工作流的总负载
    workflow.workload = sum(task.length for task in workflow.tasks)
    
    # 3. 设置工作流的截止时间
    workflow.deadline = workflow.avg_eft * (1 + rho)
    
    # 4. 计算工作流的平均松弛时间
    workflow.avg_slacktime = workflow.deadline - workflow.avg_eft

