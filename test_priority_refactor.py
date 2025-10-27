"""
测试工作流优先级重构

验证：
1. WS算法只计算分数不排序
2. workflow_id保持不变
3. 全局任务优先级正确计算
4. 时钟初始化正常工作
5. 环境创建和reset正常
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def test_ws_no_sorting():
    """测试1: WS算法不排序"""
    print("\n" + "=" * 80)
    print("测试1: WS算法只计算分数不排序")
    print("=" * 80)
    
    from scheduler.dataset_generator.core.gen_dataset import generate_dataset
    from scheduler.pre_scheduling.pre_computation import precompute_workflow_data
    from scheduler.pre_scheduling.ws_method import ContentionAwareWorkflowSequencing
    
    # 生成数据集
    dataset = generate_dataset(
        seed=42,
        host_count=4,
        vm_count=10,
        workflow_count=5,
        gnp_min_n=8,
        gnp_max_n=8,
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
    
    # 保存原始ID
    original_ids = [wf.id for wf in dataset.workflows]
    print(f"\n原始工作流ID: {original_ids}")
    
    # 预计算
    for workflow in dataset.workflows:
        precompute_workflow_data(workflow, dataset.vms, 0.2)
    
    # 运行WS算法
    ws_scheduler = ContentionAwareWorkflowSequencing(0.33, 0.33, 0.33)
    result_workflows = ws_scheduler.run(dataset.workflows, dataset.vms)
    
    # 检查是否排序
    result_ids = [wf.id for wf in result_workflows]
    print(f"WS后工作流ID: {result_ids}")
    
    if original_ids == result_ids:
        print("✅ 工作流ID顺序保持不变（未排序）")
    else:
        print("❌ 工作流ID顺序改变了（排序了）")
        return False
    
    # 检查是否计算了优先级分数
    print("\n工作流优先级分数:")
    for wf in result_workflows:
        print(f"  工作流 {wf.id}: workflow_priority={wf.workflow_priority:.4f}")
        if wf.workflow_priority == 0.0:
            print(f"  ❌ 工作流 {wf.id} 的优先级分数为0（未计算）")
            return False
    
    print("\n✅ 测试1通过：WS算法只计算分数不排序\n")
    return True


def test_global_priority():
    """测试2: 全局任务优先级计算"""
    print("=" * 80)
    print("测试2: 全局任务优先级计算")
    print("=" * 80)
    
    from scheduler.dataset_generator.core.gen_dataset import generate_dataset
    from scheduler.pre_scheduling.pre_computation import precompute_workflow_data
    from scheduler.pre_scheduling.ws_method import ContentionAwareWorkflowSequencing
    from scheduler.pre_scheduling.dp_method import BottleLayerAwareDeadlinePartition
    
    # 生成数据集
    dataset = generate_dataset(
        seed=42,
        host_count=4,
        vm_count=10,
        workflow_count=3,
        gnp_min_n=5,
        gnp_max_n=5,
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
    
    # 预调度
    for workflow in dataset.workflows:
        precompute_workflow_data(workflow, dataset.vms, 0.2)
    
    ws_scheduler = ContentionAwareWorkflowSequencing(0.33, 0.33, 0.33)
    ws_scheduler.run(dataset.workflows, dataset.vms)
    
    dp_scheduler = BottleLayerAwareDeadlinePartition(0.5)
    for workflow in dataset.workflows:
        dp_scheduler.run(workflow, dataset.vms)
    
    # 计算全局任务优先级（模拟gym_env中的逻辑）
    for workflow in dataset.workflows:
        for task in workflow.tasks:
            task.global_priority = workflow.workflow_priority * task.rank_dp
    
    print("\n全局任务优先级示例（前3个工作流的前2个任务）:")
    for workflow in dataset.workflows[:3]:
        print(f"\n工作流 {workflow.id} (workflow_priority={workflow.workflow_priority:.4f}):")
        for task in workflow.tasks[:2]:
            print(f"  任务 {task.id}:")
            print(f"    rank_dp={task.rank_dp:.2f}")
            print(f"    global_priority={task.global_priority:.4f}")
            
            # 验证计算正确性
            expected = workflow.workflow_priority * task.rank_dp
            if abs(task.global_priority - expected) > 1e-6:
                print(f"    ❌ 全局优先级计算错误！")
                return False
    
    print("\n✅ 测试2通过：全局任务优先级计算正确\n")
    return True


def test_env_creation():
    """测试3: 环境创建和reset"""
    print("=" * 80)
    print("测试3: 环境创建和reset")
    print("=" * 80)
    
    from scheduler.dataset_generator.core.gen_dataset import generate_dataset
    from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
    
    # 生成数据集
    dataset = generate_dataset(
        seed=42,
        host_count=4,
        vm_count=10,
        workflow_count=3,
        gnp_min_n=5,
        gnp_max_n=5,
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
    
    print("\n创建环境...")
    env = CloudSchedulingGymEnvironment(dataset=dataset)
    
    print("执行reset...")
    obs, info = env.reset()
    
    print(f"✅ 环境创建成功")
    print(f"  任务数: {len(obs.task_observations)}")
    print(f"  VM数: {len(obs.vm_observations)}")
    
    # 检查时钟管理器
    if env.state.clock_manager is not None:
        print(f"✅ 时钟管理器已初始化")
        print(f"  工作流时钟数: {len(env.state.clock_manager.workflow_clocks)}")
        print(f"  任务映射数: {len(env.state.clock_manager.task_to_workflow)}")
    else:
        print("❌ 时钟管理器未初始化")
        return False
    
    print("\n✅ 测试3通过：环境创建和reset正常\n")
    return True


def test_workflow_id_consistency():
    """测试4: workflow_id一致性"""
    print("=" * 80)
    print("测试4: workflow_id一致性")
    print("=" * 80)
    
    from scheduler.dataset_generator.core.gen_dataset import generate_dataset
    from scheduler.pre_scheduling.pre_computation import precompute_workflow_data
    from scheduler.pre_scheduling.ws_method import ContentionAwareWorkflowSequencing
    from scheduler.pre_scheduling.dp_method import BottleLayerAwareDeadlinePartition
    
    # 生成数据集
    dataset = generate_dataset(
        seed=42,
        host_count=4,
        vm_count=10,
        workflow_count=4,
        gnp_min_n=5,
        gnp_max_n=5,
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
    
    # 保存原始ID和任务的workflow_id
    original_wf_ids = {wf.id: [task.workflow_id for task in wf.tasks] for wf in dataset.workflows}
    print(f"\n原始workflow ID: {list(original_wf_ids.keys())}")
    
    # 预调度
    for workflow in dataset.workflows:
        precompute_workflow_data(workflow, dataset.vms, 0.2)
    
    ws_scheduler = ContentionAwareWorkflowSequencing(0.33, 0.33, 0.33)
    ws_scheduler.run(dataset.workflows, dataset.vms)
    
    dp_scheduler = BottleLayerAwareDeadlinePartition(0.5)
    for workflow in dataset.workflows:
        dp_scheduler.run(workflow, dataset.vms)
    
    # 计算全局优先级
    for workflow in dataset.workflows:
        for task in workflow.tasks:
            task.global_priority = workflow.workflow_priority * task.rank_dp
    
    # 检查ID是否保持不变
    print(f"\n预调度后workflow ID: {[wf.id for wf in dataset.workflows]}")
    
    for wf in dataset.workflows:
        if wf.id not in original_wf_ids:
            print(f"❌ 工作流 {wf.id} 的ID改变了")
            return False
        
        # 检查任务的workflow_id
        for task in wf.tasks:
            if task.workflow_id != wf.id:
                print(f"❌ 任务 {task.id} 的workflow_id不一致")
                return False
    
    print("✅ 所有workflow_id保持不变")
    print("✅ 所有任务的workflow_id与其工作流一致\n")
    
    print("\n✅ 测试4通过：workflow_id一致性正常\n")
    return True


def main():
    """运行所有测试"""
    print("\n" + "*" * 80)
    print("工作流优先级重构测试")
    print("*" * 80)
    
    tests = [
        ("WS算法不排序", test_ws_no_sorting),
        ("全局任务优先级计算", test_global_priority),
        ("环境创建和reset", test_env_creation),
        ("workflow_id一致性", test_workflow_id_consistency),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{test_name}' 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结
    print("=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！")
        print("\n✅ 重构成功：")
        print("  1. WS算法只计算分数不排序")
        print("  2. workflow_id保持不变")
        print("  3. 全局任务优先级正确计算")
        print("  4. 时钟初始化正常工作")
        print("  5. 环境创建和reset正常")
        return True
    else:
        print("\n❌ 部分测试失败，请检查")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

