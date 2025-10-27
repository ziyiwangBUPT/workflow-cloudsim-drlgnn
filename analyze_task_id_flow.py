"""
分析Task ID流程

详细追踪Task ID在整个流程中的变化
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scheduler.dataset_generator.core.gen_dataset import generate_dataset
from scheduler.rl_model.core.types import TaskDto
from scheduler.rl_model.core.utils.task_mapper import TaskMapper
from scheduler.pre_scheduling.pre_computation import precompute_workflow_data
from scheduler.pre_scheduling.ws_method import ContentionAwareWorkflowSequencing
from scheduler.pre_scheduling.dp_method import BottleLayerAwareDeadlinePartition


def analyze_full_flow():
    """完整流程分析"""
    print("\n" + "=" * 80)
    print("Task ID完整流程分析")
    print("=" * 80)
    
    # 步骤1：生成数据集
    print("\n步骤1: 生成数据集")
    print("-" * 60)
    dataset = generate_dataset(
        seed=42,
        host_count=4,
        vm_count=10,
        workflow_count=5,
        gnp_min_n=10,
        gnp_max_n=10,
        max_memory_gb=16,
        min_cpu_speed_mips=1000,
        max_cpu_speed_mips=3000,
        dag_method='gnp',
        task_length_dist='uniform',
        min_task_length=10000,
        max_task_length=100000,
        task_arrival='static',
        arrival_rate=1.0
    )
    
    print(f"生成了 {len(dataset.workflows)} 个工作流")
    print("\n原始工作流顺序和任务数量:")
    for workflow in dataset.workflows:
        print(f"  工作流 {workflow.id}: {len(workflow.tasks)} 个任务")
    
    # 步骤2：预调度
    print("\n步骤2: 预调度（WS和DP）")
    print("-" * 60)
    
    ws_scheduler = ContentionAwareWorkflowSequencing(alpha1=0.33, alpha2=0.33, alpha3=0.33)
    dp_scheduler = BottleLayerAwareDeadlinePartition(beta=0.5)
    rho = 0.2
    
    # 预计算
    for workflow in dataset.workflows:
        precompute_workflow_data(workflow, dataset.vms, rho)
    
    # 工作流排序
    print("\nWS算法排序前的工作流ID:")
    print(f"  {[wf.id for wf in dataset.workflows]}")
    
    sorted_workflows = ws_scheduler.run(dataset.workflows, dataset.vms)
    
    print("\nWS算法排序后的工作流ID:")
    print(f"  {[wf.id for wf in sorted_workflows]}")
    
    # ⚠️ 关键：工作流顺序改变了！
    order_changed = sorted_workflows != dataset.workflows
    print(f"\n⚠️ 工作流顺序是否改变: {order_changed}")
    
    # DP算法
    for workflow in sorted_workflows:
        dp_scheduler.run(workflow, dataset.vms)
    
    # 模拟gym_env中的操作
    dataset.workflows = sorted_workflows
    
    # 步骤3：转换为TaskDto
    print("\n步骤3: 转换为TaskDto")
    print("-" * 60)
    
    tasks = [TaskDto.from_task(task) for workflow in dataset.workflows for task in workflow.tasks]
    print(f"转换了 {len(tasks)} 个任务")
    
    print("\n前10个TaskDto:")
    for i, task in enumerate(tasks[:10]):
        print(f"  索引 {i}: 原始ID={task.id}, workflow_id={task.workflow_id}")
    
    # 步骤4：TaskMapper映射
    print("\n步骤4: TaskMapper映射")
    print("-" * 60)
    
    task_mapper = TaskMapper(tasks)
    print(f"_task_counts_cum: {task_mapper._task_counts_cum}")
    
    mapped_tasks = task_mapper.map_tasks()
    print(f"映射后任务数: {len(mapped_tasks)}")
    
    print("\n映射后的前15个任务:")
    for i, task in enumerate(mapped_tasks[:15]):
        print(f"  索引 {i}: ID={task.id}, workflow_id={task.workflow_id}")
    
    # 步骤5：检查ID连续性
    print("\n步骤5: 检查ID连续性")
    print("-" * 60)
    
    errors = []
    for i, task in enumerate(mapped_tasks):
        if task.id != i:
            errors.append((i, task.id))
    
    if errors:
        print(f"❌ 发现 {len(errors)} 个ID不匹配:")
        for idx, (i, task_id) in enumerate(errors[:10]):
            print(f"  索引 {i}: ID={task_id} (期望{i})")
            if idx == 0:
                # 显示前后任务
                print(f"    上一个任务: 索引{i-1}, ID={mapped_tasks[i-1].id if i>0 else 'N/A'}")
                print(f"    下一个任务: 索引{i+1}, ID={mapped_tasks[i+1].id if i+1<len(mapped_tasks) else 'N/A'}")
    else:
        print("✓ 所有任务ID连续")
    
    # 分析原因
    print("\n分析: 为什么会出现ID不匹配？")
    print("-" * 60)
    
    if order_changed:
        print("⚠️ 工作流顺序在WS算法后改变了！")
        print("\n问题根源：")
        print("  1. TaskMapper基于原始workflow_id计算映射ID")
        print("  2. 但工作流在WS算法后被重新排序")
        print("  3. TaskDto按排序后的顺序展开")
        print("  4. 导致mapped_tasks的顺序与预期的ID顺序不一致")
        
        print("\n举例:")
        if len(sorted_workflows) >= 3:
            print(f"  排序后第1个工作流(索引0)的原始ID: {sorted_workflows[0].id}")
            print(f"  排序后第2个工作流(索引1)的原始ID: {sorted_workflows[1].id}")
            print(f"  排序后第3个工作流(索引2)的原始ID: {sorted_workflows[2].id}")
            
            first_wf_id = sorted_workflows[0].id
            second_wf_id = sorted_workflows[1].id
            
            print(f"\n  第1个工作流(原始ID={first_wf_id})的第1个任务:")
            print(f"    在mapped_tasks中的索引: 1")
            print(f"    其mapped ID计算: _task_counts_cum[{first_wf_id}] + 0 + 1")
            print(f"                      = {task_mapper._task_counts_cum[first_wf_id]} + 0 + 1")
            print(f"                      = {task_mapper._task_counts_cum[first_wf_id] + 1}")
            
            print(f"\n  第2个工作流(原始ID={second_wf_id})的第1个任务:")
            first_wf_task_count = len(sorted_workflows[0].tasks)
            expected_index = 1 + first_wf_task_count
            print(f"    在mapped_tasks中的预期索引: {expected_index}")
            print(f"    其mapped ID计算: _task_counts_cum[{second_wf_id}] + 0 + 1")
            print(f"                      = {task_mapper._task_counts_cum[second_wf_id]} + 0 + 1")
            print(f"                      = {task_mapper._task_counts_cum[second_wf_id] + 1}")
            
            print(f"\n  ❌ 不匹配: 索引应该是{expected_index}，但ID是{task_mapper._task_counts_cum[second_wf_id] + 1}")


if __name__ == "__main__":
    try:
        analyze_full_flow()
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

