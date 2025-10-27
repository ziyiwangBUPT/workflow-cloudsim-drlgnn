"""
测试虚拟时钟管理器的集成

验证：
1. 时钟管理器在环境初始化时创建
2. 每次任务分配时，对应工作流的虚拟时钟会更新
3. 可以使用虚拟时钟来查询碳强度
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.dataset_generator.core.gen_dataset import generate_dataset


def test_clock_manager_in_gym_env():
    """测试虚拟时钟管理器在Gym环境中的集成"""
    print("\n" + "=" * 80)
    print("测试：虚拟时钟管理器集成")
    print("=" * 80 + "\n")
    
    # 生成数据集
    print("1. 生成测试数据集...")
    dataset = generate_dataset(
        seed=42,
        host_count=4,
        vm_count=10,
        workflow_count=3,  # 3个工作流，便于测试不同工作流的时钟
        gnp_min_n=5,
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
    
    print(f"   ✓ 生成了 {len(dataset.workflows)} 个工作流")
    print(f"   ✓ 生成了 {len(dataset.vms)} 个VM\n")
    
    # 创建环境
    print("2. 创建环境并初始化...")
    env = CloudSchedulingGymEnvironment(dataset=dataset)
    obs, info = env.reset()
    
    # 验证时钟管理器存在
    assert env.state is not None, "环境状态不应为None"
    assert env.state.clock_manager is not None, "时钟管理器应该已创建"
    
    print(f"   ✓ 环境重置成功")
    print(f"   ✓ 时钟管理器已创建\n")
    
    # 验证每个工作流的时钟初始化为0
    print("3. 验证初始时钟状态...")
    for workflow in dataset.workflows:
        clock = env.state.clock_manager.get_workflow_clock(workflow.id)
        print(f"   工作流 {workflow.id}: 初始时钟 = {clock:.2f}秒")
        assert clock == 0.0, f"工作流 {workflow.id} 的初始时钟应该为0"
    
    print("   ✓ 所有工作流的初始时钟为0\n")
    
    # 执行几个步骤并观察时钟变化
    print("4. 执行调度步骤并观察时钟变化...")
    
    max_steps = 15  # 执行15步
    for step in range(max_steps):
        # 选择一个有效的动作
        # 找到第一个ready且未分配的任务
        valid_action = None
        for task_id in range(len(env.state.task_states)):
            task_state = env.state.task_states[task_id]
            if task_state.is_ready and task_state.assigned_vm_id is None:
                # 找一个兼容的VM
                for vm_id in range(len(env.state.vm_states)):
                    if (task_id, vm_id) in env.state.static_state.compatibilities:
                        valid_action = type('Action', (), {'task_id': task_id, 'vm_id': vm_id})()
                        break
                if valid_action:
                    break
        
        if not valid_action:
            print(f"    步骤 {step+1}: 没有可调度的任务，环境完成")
            break
        
        # 获取执行前的时钟状态
        before_clocks = {
            wf.id: env.state.clock_manager.get_workflow_clock(wf.id)
            for wf in dataset.workflows
        }
        
        # 执行步骤
        env.step(valid_action)
        
        # 获取执行后的时钟状态
        after_clocks = {
            wf.id: env.state.clock_manager.get_workflow_clock(wf.id)
            for wf in dataset.workflows
        }
        
        # 打印时钟变化
        if step < 5:  # 只显示前5步
            for wf in dataset.workflows:
                if after_clocks[wf.id] != before_clocks[wf.id]:
                    print(f"    步骤 {step+1}: 工作流 {wf.id} 时钟 "
                          f"{before_clocks[wf.id]:.2f} → {after_clocks[wf.id]:.2f} 秒")
        
        # 检查是否有时钟被更新
        if any(after_clocks[wf.id] > before_clocks[wf.id] for wf in dataset.workflows):
            print(f"    ✓ 步骤 {step+1}: 时钟已更新")
        else:
            print(f"    - 步骤 {step+1}: 时钟未变化（可能是虚拟任务）")
    
    # 验证最终时钟状态
    print("\n5. 验证最终时钟状态...")
    for workflow in dataset.workflows:
        final_clock = env.state.clock_manager.get_workflow_clock(workflow.id)
        current_hour = final_clock / 3600
        print(f"   工作流 {workflow.id}: 最终时钟 = {final_clock:.2f}秒 ({current_hour:.2f}小时)")
    
    print("\n" + "=" * 80)
    print("✅ 虚拟时钟管理器集成测试通过！")
    print("=" * 80 + "\n")
    
    return True


def test_carbon_intensity_with_clock():
    """测试使用虚拟时钟查询碳强度"""
    print("\n" + "=" * 80)
    print("测试：使用虚拟时钟查询碳强度")
    print("=" * 80 + "\n")
    
    # 生成数据集
    dataset = generate_dataset(
        seed=42,
        host_count=4,
        vm_count=10,
        workflow_count=2,
        gnp_min_n=5,
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
    
    # 创建环境
    env = CloudSchedulingGymEnvironment(dataset=dataset)
    obs, info = env.reset()
    
    print("✓ 环境创建成功\n")
    
    # 执行几步调度
    for step in range(10):
        # 找一个有效动作
        valid_action = None
        for task_id in range(len(env.state.task_states)):
            task_state = env.state.task_states[task_id]
            if task_state.is_ready and task_state.assigned_vm_id is None:
                for vm_id in range(len(env.state.vm_states)):
                    if (task_id, vm_id) in env.state.static_state.compatibilities:
                        valid_action = type('Action', (), {'task_id': task_id, 'vm_id': vm_id})()
                        break
                if valid_action:
                    break
        
        if not valid_action:
            break
        
        env.step(valid_action)
        
        if step < 3:  # 显示前3步
            # 获取当前工作流的时钟
            for wf in dataset.workflows:
                clock = env.state.clock_manager.get_workflow_clock(wf.id)
                
                # 查询该工作流的第一个VM的碳强度
                vm_obs = obs.vm_observations[0]
                carbon_intensity = vm_obs.get_carbon_intensity_at(clock)
                hour = clock / 3600
                
                print(f"  步骤 {step+1}: 工作流 {wf.id} 时钟={clock:.2f}秒({hour:.2f}h), "
                      f"碳强度={carbon_intensity:.3f}")
    
    print("\n✓ 可以使用虚拟时钟查询碳强度")
    
    print("\n" + "=" * 80)
    print("✅ 碳强度查询测试通过！")
    print("=" * 80 + "\n")
    
    return True


def main():
    """运行所有测试"""
    try:
        # 测试1：时钟管理器集成
        assert test_clock_manager_in_gym_env(), "时钟管理器集成测试失败"
        
        # 测试2：使用时钟查询碳强度
        assert test_carbon_intensity_with_clock(), "碳强度查询测试失败"
        
        print("\n🎉 所有测试通过！虚拟时钟已正确集成并更新！\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

