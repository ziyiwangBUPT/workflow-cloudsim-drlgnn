"""
预调度功能测试脚本
用于验证 WS 和 DP 算法是否正确集成
"""
import sys
from pathlib import Path

# 确保能够导入 scheduler 模块
sys.path.insert(0, str(Path(__file__).parent))

from scheduler.dataset_generator.core.gen_dataset import generate_dataset
from scheduler.pre_scheduling.pre_computation import precompute_workflow_data
from scheduler.pre_scheduling.ws_method import ContentionAwareWorkflowSequencing
from scheduler.pre_scheduling.dp_method import BottleLayerAwareDeadlinePartition


def test_pre_scheduling():
    """测试预调度功能"""
    
    print("=" * 80)
    print("开始测试预调度功能")
    print("=" * 80)
    
    # 1. 生成测试数据集
    print("\n1. 生成测试数据集...")
    dataset = generate_dataset(
        seed=42,
        host_count=3,
        vm_count=10,
        workflow_count=5,
        gnp_min_n=5,
        gnp_max_n=10,
        max_memory_gb=16,
        min_cpu_speed_mips=1000,
        max_cpu_speed_mips=3000,
        dag_method='gnp',
        task_length_dist='uniform',
        min_task_length=100,
        max_task_length=1000,
        task_arrival='static',  # 修正：使用 'static' 而不是 'fixed'
        arrival_rate=1.0
    )
    
    print(f"   - 生成了 {len(dataset.workflows)} 个工作流")
    print(f"   - 生成了 {len(dataset.vms)} 个虚拟机")
    print(f"   - 总任务数: {sum(len(wf.tasks) for wf in dataset.workflows)}")
    
    # 2. 预计算每个工作流的数据
    print("\n2. 预计算工作流数据...")
    rho = 0.2
    for i, workflow in enumerate(dataset.workflows):
        precompute_workflow_data(workflow, dataset.vms, rho)
        print(f"   - 工作流 {i}: avg_eft={workflow.avg_eft:.2f}, "
              f"workload={workflow.workload:.2f}, "
              f"deadline={workflow.deadline:.2f}, "
              f"avg_slacktime={workflow.avg_slacktime:.2f}")
    
    # 3. 工作流排序 (WS)
    print("\n3. 运行工作流排序 (WS) 算法...")
    ws_scheduler = ContentionAwareWorkflowSequencing(alpha1=0.33, alpha2=0.33, alpha3=0.33)
    sorted_workflows = ws_scheduler.run(dataset.workflows, dataset.vms)
    print(f"   - 工作流已排序，顺序: {[wf.id for wf in sorted_workflows]}")
    
    # 4. 截止时间划分 (DP)
    print("\n4. 运行截止时间划分 (DP) 算法...")
    dp_scheduler = BottleLayerAwareDeadlinePartition(beta=0.5)
    for workflow in sorted_workflows:
        dp_scheduler.run(workflow, dataset.vms)
    
    # 5. 验证结果
    print("\n5. 验证每个任务的属性...")
    all_tasks_have_attributes = True
    for workflow in sorted_workflows:
        for task in workflow.tasks:
            if task.rank_dp == 0.0 or task.deadline == 0.0:
                all_tasks_have_attributes = False
                print(f"   [警告] 工作流 {workflow.id} 的任务 {task.id} 未正确设置属性")
    
    if all_tasks_have_attributes:
        print("   ✓ 所有任务都已正确设置 rank_dp 和 deadline 属性")
    
    # 6. 显示示例任务的详细信息
    print("\n6. 示例任务详情（第一个工作流的前3个任务）:")
    first_workflow = sorted_workflows[0]
    for i, task in enumerate(first_workflow.tasks[:3]):
        print(f"   任务 {task.id}:")
        print(f"      - length: {task.length}")
        print(f"      - avg_est: {task.avg_est:.2f}")
        print(f"      - avg_eft: {task.avg_eft:.2f}")
        print(f"      - rank_dp: {task.rank_dp:.2f}")
        print(f"      - deadline: {task.deadline:.2f}")
        print(f"      - parent_ids: {task.parent_ids}")
        print(f"      - child_ids: {task.child_ids}")
    
    print("\n" + "=" * 80)
    print("预调度功能测试完成！")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        success = test_pre_scheduling()
        if success:
            print("\n✓ 所有测试通过！")
            sys.exit(0)
        else:
            print("\n✗ 测试失败！")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

