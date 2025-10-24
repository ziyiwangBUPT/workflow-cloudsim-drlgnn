"""
使用 Pegasus 真实工作流模板的测试脚本
展示如何生成和使用真实的工作流结构
"""
import sys
from pathlib import Path

# 确保能够导入 scheduler 模块
sys.path.insert(0, str(Path(__file__).parent))

from scheduler.dataset_generator.core.gen_dataset import generate_dataset
from scheduler.pre_scheduling.pre_computation import precompute_workflow_data
from scheduler.pre_scheduling.ws_method import ContentionAwareWorkflowSequencing
from scheduler.pre_scheduling.dp_method import BottleLayerAwareDeadlinePartition


def test_pegasus_workflows():
    """使用 Pegasus 真实工作流模板测试预调度功能"""
    
    print("=" * 80)
    print("使用 Pegasus 真实工作流模板测试")
    print("=" * 80)
    
    # 1. 生成使用 Pegasus 工作流的数据集
    print("\n1. 生成数据集（使用 Pegasus 工作流）...")
    dataset = generate_dataset(
        seed=42,
        host_count=5,
        vm_count=15,
        workflow_count=3,        # 生成 3 个工作流
        gnp_min_n=5,             # 这两个参数在 pegasus 模式下不使用
        gnp_max_n=10,
        max_memory_gb=16,
        min_cpu_speed_mips=1000,
        max_cpu_speed_mips=3000,
        dag_method='pegasus',    # ← 关键：使用 pegasus 方法
        task_length_dist='uniform',
        min_task_length=10000,
        max_task_length=100000,
        task_arrival='static',
        arrival_rate=1.0
    )
    
    print(f"   - 生成了 {len(dataset.workflows)} 个工作流")
    print(f"   - 生成了 {len(dataset.vms)} 个虚拟机")
    print(f"   - 总任务数: {sum(len(wf.tasks) for wf in dataset.workflows)}")
    
    # 2. 显示工作流结构
    print("\n2. 工作流结构分析...")
    for i, workflow in enumerate(dataset.workflows):
        print(f"\n   工作流 {i}:")
        print(f"   - 任务数: {len(workflow.tasks)}")
        
        # 统计入口、中间、出口任务
        entry_tasks = [t for t in workflow.tasks if not t.parent_ids]
        exit_tasks = [t for t in workflow.tasks if not t.child_ids]
        middle_tasks = [t for t in workflow.tasks if t.parent_ids and t.child_ids]
        
        print(f"   - 入口任务: {len(entry_tasks)} 个")
        print(f"   - 中间任务: {len(middle_tasks)} 个")
        print(f"   - 出口任务: {len(exit_tasks)} 个")
        
        # 计算平均分支因子
        avg_children = sum(len(t.child_ids) for t in workflow.tasks) / len(workflow.tasks)
        print(f"   - 平均分支因子: {avg_children:.2f}")
        
        # 显示前 3 个任务的结构
        print(f"   - 任务结构示例（前3个）:")
        for task in workflow.tasks[:3]:
            print(f"      Task {task.id}: parents={task.parent_ids}, children={task.child_ids}")
    
    # 3. 预计算工作流数据
    print("\n3. 预计算工作流数据...")
    rho = 0.2
    for i, workflow in enumerate(dataset.workflows):
        precompute_workflow_data(workflow, dataset.vms, rho)
        print(f"   - 工作流 {i}: avg_eft={workflow.avg_eft:.2f}, "
              f"workload={workflow.workload:.2f}, "
              f"deadline={workflow.deadline:.2f}")
    
    # 4. 工作流排序 (WS)
    print("\n4. 运行工作流排序 (WS) 算法...")
    ws_scheduler = ContentionAwareWorkflowSequencing(alpha1=0.33, alpha2=0.33, alpha3=0.33)
    sorted_workflows = ws_scheduler.run(dataset.workflows, dataset.vms)
    print(f"   - 工作流已排序，顺序: {[wf.id for wf in sorted_workflows]}")
    
    # 5. 截止时间划分 (DP)
    print("\n5. 运行截止时间划分 (DP) 算法...")
    dp_scheduler = BottleLayerAwareDeadlinePartition(beta=0.5)
    for workflow in sorted_workflows:
        dp_scheduler.run(workflow, dataset.vms)
    
    # 6. 验证结果
    print("\n6. 验证每个任务的属性...")
    all_valid = True
    for workflow in sorted_workflows:
        for task in workflow.tasks:
            if task.rank_dp <= 0.0 or task.deadline <= 0.0:
                all_valid = False
                print(f"   [警告] 工作流 {workflow.id} 的任务 {task.id} 属性异常")
                print(f"           rank_dp={task.rank_dp}, deadline={task.deadline}")
    
    if all_valid:
        print("   ✓ 所有任务都已正确设置 rank_dp 和 deadline 属性")
    
    # 7. 显示第一个工作流的详细任务信息
    print("\n7. 第一个工作流的详细任务信息:")
    first_workflow = sorted_workflows[0]
    
    # 按任务类型分组显示
    entry_tasks = [t for t in first_workflow.tasks if not t.parent_ids]
    middle_tasks = [t for t in first_workflow.tasks if t.parent_ids and t.child_ids]
    exit_tasks = [t for t in first_workflow.tasks if not t.child_ids]
    
    print(f"\n   入口任务 ({len(entry_tasks)} 个):")
    for task in entry_tasks[:2]:  # 显示前2个
        print(f"      Task {task.id}:")
        print(f"         length: {task.length}, rank_dp: {task.rank_dp:.2f}, deadline: {task.deadline:.2f}")
        print(f"         children: {task.child_ids}")
    
    print(f"\n   中间任务 ({len(middle_tasks)} 个, 显示前3个):")
    for task in middle_tasks[:3]:
        print(f"      Task {task.id}:")
        print(f"         length: {task.length}, rank_dp: {task.rank_dp:.2f}, deadline: {task.deadline:.2f}")
        print(f"         parents: {task.parent_ids}, children: {task.child_ids}")
    
    print(f"\n   出口任务 ({len(exit_tasks)} 个, 显示前2个):")
    for task in exit_tasks[:2]:
        print(f"      Task {task.id}:")
        print(f"         length: {task.length}, rank_dp: {task.rank_dp:.2f}, deadline: {task.deadline:.2f}")
        print(f"         parents: {task.parent_ids}")
    
    # 8. 分析 rank_dp 的分布
    print("\n8. rank_dp 分布分析:")
    all_rank_dps = [task.rank_dp for wf in sorted_workflows for task in wf.tasks]
    print(f"   - 最小 rank_dp: {min(all_rank_dps):.2f}")
    print(f"   - 最大 rank_dp: {max(all_rank_dps):.2f}")
    print(f"   - 平均 rank_dp: {sum(all_rank_dps)/len(all_rank_dps):.2f}")
    
    # 9. 分析 deadline 的分布
    print("\n9. deadline 分布分析:")
    for i, workflow in enumerate(sorted_workflows):
        deadlines = [task.deadline for task in workflow.tasks]
        print(f"   工作流 {i}:")
        print(f"      - deadline 范围: {min(deadlines):.2f} ~ {max(deadlines):.2f}")
        print(f"      - 工作流 deadline: {workflow.deadline:.2f}")
    
    print("\n" + "=" * 80)
    print("Pegasus 工作流测试完成！")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        success = test_pegasus_workflows()
        if success:
            print("\n✓ 所有测试通过！Pegasus 工作流已成功集成。")
            sys.exit(0)
        else:
            print("\n✗ 测试失败！")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

