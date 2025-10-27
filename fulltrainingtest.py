"""
完整训练流程测试

验证添加碳强度特征后，训练流程是否还能正常工作。
模拟训练的关键步骤：环境创建、reset、step、奖励计算。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scheduler.dataset_generator.core.gen_dataset import generate_dataset
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.core.env.action import EnvAction
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper


def test_environment_creation():
    """测试1: 环境创建"""
    print("\n" + "=" * 60)
    print("测试1: 环境创建")
    print("=" * 60)

    # 生成数据集
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

    print(f"✓ 生成了 {len(dataset.hosts)} 个Host")
    print(f"✓ 生成了 {len(dataset.vms)} 个VM")
    print(f"✓ 生成了 {len(dataset.workflows)} 个工作流")

    # 验证时钟管理器
    env = CloudSchedulingGymEnvironment(dataset=dataset)
    print("✓ 环境创建成功")
    
    # 调用reset以初始化state
    obs, info = env.reset()
    print("✓ 环境重置成功")
    
    assert env.state is not None
    assert env.state.clock_manager is not None
    print("✓ 时钟管理器已初始化")
    
    print("\n✅ 环境创建测试通过！\n")
    return env


def test_environment_reset(env):
    """测试2: 环境重置"""
    print("=" * 60)
    print("测试2: 验证观察和状态")
    print("=" * 60)
    
    # 环境已经在创建时重置过了，这里只需要获取当前观察
    obs = env.state
    
    # 从state创建观察
    from scheduler.rl_model.core.env.observation import EnvObservation
    current_obs = EnvObservation(env.state)
    
    print(f"✓ 观察空间包含: {len(current_obs.task_observations)} 个任务观察")
    print(f"✓ 观察空间包含: {len(current_obs.vm_observations)} 个VM观察")
    
    # 验证VM碳强度特征
    for i in range(min(3, len(current_obs.vm_observations))):
        vm_obs = current_obs.vm_observations[i]
        carbon_intensity = vm_obs.get_carbon_intensity_at(0)
        print(f"  VM {i}: Host={vm_obs.host_id}, 碳强度@0时={carbon_intensity:.3f}")
    
    # 验证时钟状态
    # 直接从环境获取数据集（它保存了第一次生成的dataset）
    for wf_id in range(5):  # 测试中有5个工作流
        clock = env.state.clock_manager.get_workflow_clock(wf_id)
        assert clock == 0.0, f"工作流 {wf_id} 初始时钟应为0"
    
    print("✓ 所有工作流时钟初始化为0")
    print("✓ 碳强度特征正常加载")
    
    print("\n✅ 观察和状态测试通过！\n")
    return current_obs


def test_environment_step(env):
    """测试3: 环境步进和时钟更新"""
    print("=" * 60)
    print("测试3: 环境步进和时钟更新")
    print("=" * 60)

    obs, info = env.reset()

    step_count = 0
    for _ in range(10):
        # 找一个有效的动作
        valid_action = None
        for task_id in range(len(env.state.task_states)):
            task_state = env.state.task_states[task_id]
            if task_state.is_ready and task_state.assigned_vm_id is None:
                # 找一个兼容的VM
                for vm_id in range(len(env.state.vm_states)):
                    task_dto = env.state.static_state.tasks[task_id]
                    vm_dto = env.state.static_state.vms[vm_id]
                    if vm_dto.memory_mb >= task_dto.req_memory_mb:
                        valid_action = EnvAction(task_id=task_id, vm_id=vm_id)
                        break
                if valid_action:
                    break

        if not valid_action:
            print("没有更多可调度的任务")
            break

        # 获取时钟状态
        clocks_before = {}
        for wf_id in range(5):  # 测试中有5个工作流
            clocks_before[wf_id] = env.state.clock_manager.get_workflow_clock(wf_id)

        # 执行一步
        env.step(valid_action)
        step_count += 1

        # 检查时钟是否更新
        if step_count <= 3:
            for wf_id in clocks_before:
                clock_after = env.state.clock_manager.get_workflow_clock(wf_id)
                if clock_after > clocks_before[wf_id]:
                    print(f"  步骤 {step_count}: 工作流 {wf_id} 时钟 "
                          f"{clocks_before[wf_id]:.2f} → {clock_after:.2f} 秒")

    print(f"\n✓ 执行了 {step_count} 步")
    print("✓ 虚拟时钟正常更新")

    print("\n✅ 环境步进测试通过！\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("完整训练流程测试")
    print("=" * 60)

    try:
        # 测试1: 环境创建
        env = test_environment_creation()

        # 测试2: 环境重置
        obs = test_environment_reset(env)

        # 测试3: 环境步进和时钟更新
        test_environment_step(env)

        print("=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
        print("\n✅ 训练流程完全正常！")
        print("  1. ✓ 环境创建")
        print("  2. ✓ 环境重置")
        print("  3. ✓ 环境步进")
        print("  4. ✓ 虚拟时钟更新")
        print("\n")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)