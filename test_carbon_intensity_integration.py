"""
碳强度特征集成测试脚本

验证碳强度特征是否正确集成到项目中。
"""
import sys
from pathlib import Path

# 确保能够导入 scheduler 模块
sys.path.insert(0, str(Path(__file__).parent))

from scheduler.config.carbon_intensity import (
    CARBON_INTENSITY_DATA,
    FIXED_NUM_HOSTS,
    get_carbon_intensity_at_time,
    calculate_carbon_cost
)
from scheduler.dataset_generator.core.gen_dataset import generate_dataset
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper


def test_carbon_intensity_data():
    """测试碳强度数据配置"""
    print("=" * 80)
    print("测试1: 碳强度数据配置")
    print("=" * 80)
    
    print(f"✓ 固定Host数量: {FIXED_NUM_HOSTS}")
    print(f"✓ 碳强度数据形状: {len(CARBON_INTENSITY_DATA)} hosts × {len(CARBON_INTENSITY_DATA[0])} hours")
    
    # 测试每个Host的碳强度曲线
    for host_id in range(FIXED_NUM_HOSTS):
        curve = CARBON_INTENSITY_DATA[host_id]
        print(f"\n  Host {host_id} 碳强度曲线（24小时）:")
        print(f"    最小值: {min(curve):.3f}")
        print(f"    最大值: {max(curve):.3f}")
        print(f"    平均值: {sum(curve)/len(curve):.3f}")
        print(f"    前6小时: {curve[:6]}")
    
    # 测试获取特定时间的碳强度
    print(f"\n✓ 测试时间点碳强度查询:")
    for hour in [0, 6, 12, 18]:
        time_seconds = hour * 3600
        carbon_intensity = get_carbon_intensity_at_time(host_id=0, time_seconds=time_seconds)
        print(f"    Host 0, {hour}:00 时 → 碳强度 = {carbon_intensity:.3f}")
    
    print("\n✅ 碳强度数据配置测试通过！\n")
    return True


def test_host_generation():
    """测试Host生成逻辑"""
    print("=" * 80)
    print("测试2: Host生成逻辑")
    print("=" * 80)
    
    # 测试：请求不同数量的Host，应该始终生成4个
    test_counts = [2, 4, 5, 10]
    
    for idx, requested_count in enumerate(test_counts):
        # 每次使用不同的seed避免任务ID冲突
        dataset = generate_dataset(
            seed=42 + idx,  # 使用不同的seed
            host_count=requested_count,  # 请求的数量
            vm_count=15,
            workflow_count=2,
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
        
        print(f"\n  请求 {requested_count} 个Host → 实际生成 {len(dataset.hosts)} 个Host")
        assert len(dataset.hosts) == FIXED_NUM_HOSTS, f"Host数量错误！期望{FIXED_NUM_HOSTS}，实际{len(dataset.hosts)}"
        
        # 验证每个Host都有碳强度曲线
        for host in dataset.hosts:
            assert host.carbon_intensity_curve is not None, f"Host {host.id} 缺少碳强度曲线！"
            assert len(host.carbon_intensity_curve) == 24, f"Host {host.id} 碳强度曲线长度错误！"
            print(f"    ✓ Host {host.id}: 有效的碳强度曲线（24小时）")
    
    print("\n✅ Host生成逻辑测试通过！\n")
    return True


def test_vm_carbon_features():
    """测试VM碳强度特征"""
    print("=" * 80)
    print("测试3: VM碳强度特征")
    print("=" * 80)
    
    # 生成数据集 - 使用不同的seed避免与前一个测试冲突
    dataset = generate_dataset(
        seed=100,  # 使用不同的seed
        host_count=4,
        vm_count=12,  # 测试多个VM
        workflow_count=2,
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
    
    print(f"\n  生成了 {len(dataset.vms)} 个VM")
    
    # 创建环境并重置
    env = CloudSchedulingGymEnvironment(dataset=dataset)
    obs, info = env.reset()
    
    print(f"  环境重置成功，任务数: {len(obs.task_observations)}, VM数: {len(obs.vm_observations)}")
    
    # 验证每个VM都有碳强度特征
    print(f"\n  验证VM碳强度特征:")
    for i, vm_obs in enumerate(obs.vm_observations[:4]):  # 只显示前4个
        carbon_intensity = vm_obs.get_carbon_intensity_at(0)
        print(f"    VM {i}: Host ID = {vm_obs.host_id}, 碳强度@0时 = {carbon_intensity:.3f}")
        assert vm_obs.host_carbon_intensity_curve is not None, f"VM {i} 缺少碳强度曲线！"
    
    print("\n✅ VM碳强度特征测试通过！\n")
    return True


def test_gnn_features():
    """测试GNN特征映射"""
    print("=" * 80)
    print("测试4: GNN特征映射")
    print("=" * 80)
    
    # 生成数据集 - 使用不同的seed
    dataset = generate_dataset(
        seed=200,  # 使用不同的seed
        host_count=4,
        vm_count=10,
        workflow_count=2,
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
    
    # 创建包装的环境
    env = CloudSchedulingGymEnvironment(dataset=dataset)
    wrapped_env = GinAgentWrapper(env)
    
    print(f"  观察空间形状: {wrapped_env.observation_space.shape}")
    print(f"  动作空间大小: {wrapped_env.action_space.n}")
    
    # 重置环境
    obs, info = wrapped_env.reset()
    print(f"\n  ✓ 环境重置成功")
    print(f"  ✓ 观察向量长度: {len(obs)}")
    
    # 执行几个步骤
    print(f"\n  执行5个调度步骤:")
    for step in range(5):
        action = wrapped_env.action_space.sample()
        obs, reward, done, truncated, info = wrapped_env.step(action)
        print(f"    步骤 {step+1}: 动作={action}, 奖励={reward:.4f}, 完成={done}")
        
        if done:
            print(f"    环境已完成！")
            break
    
    print("\n✅ GNN特征映射测试通过！\n")
    return True


def test_carbon_cost_calculation():
    """测试碳成本计算"""
    print("=" * 80)
    print("测试5: 碳成本计算")
    print("=" * 80)
    
    # 生成数据集 - 使用不同的seed
    dataset = generate_dataset(
        seed=300,  # 使用不同的seed
        host_count=4,
        vm_count=10,
        workflow_count=2,
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
    
    # 创建环境
    env = CloudSchedulingGymEnvironment(dataset=dataset)
    wrapped_env = GinAgentWrapper(env)
    
    # 必须通过wrapped_env来reset，这样才能初始化prev_obs
    obs_wrapped, info = wrapped_env.reset()
    
    # 获取初始观察（EnvObservation）用于显示能耗和碳成本
    # wrapper.prev_obs 在 reset() 后已初始化为 EnvObservation
    from scheduler.rl_model.core.env.observation import EnvObservation
    obs_env: EnvObservation = wrapped_env.prev_obs
    
    print(f"  初始状态:")
    print(f"    总能耗: {obs_env.energy_consumption():.2f}")
    print(f"    总碳成本: {obs_env.carbon_cost():.2f}")
    
    # 执行10个步骤
    print(f"\n  执行10个调度步骤并跟踪碳成本:")
    for step in range(10):
        action = wrapped_env.action_space.sample()
        obs_wrapped, reward, done, truncated, info = wrapped_env.step(action)
        
        # 获取当前状态的观察（EnvObservation 由 wrapper 维护）
        obs_env = wrapped_env.prev_obs
        
        if step % 3 == 0:  # 每3步显示一次
            energy = obs_env.energy_consumption()
            carbon_cost = obs_env.carbon_cost()
            print(f"    步骤 {step+1}: 能耗={energy:.2f}, 碳成本={carbon_cost:.2f}")
        
        if done:
            break
    
    # 最终状态
    print(f"\n  最终状态:")
    obs_env = wrapped_env.prev_obs
    print(f"    总能耗: {obs_env.energy_consumption():.2f}")
    print(f"    总碳成本: {obs_env.carbon_cost():.2f}")
    print(f"    makespan: {obs_env.makespan():.2f}")
    
    print("\n✅ 碳成本计算测试通过！\n")
    return True


def main():
    """运行所有测试"""
    print("\n")
    print("*" * 80)
    print("碳强度特征集成测试")
    print("*" * 80)
    print("\n")
    
    try:
        # 测试1：碳强度数据配置
        assert test_carbon_intensity_data(), "碳强度数据配置测试失败"
        
        # 测试2：Host生成逻辑
        assert test_host_generation(), "Host生成逻辑测试失败"
        
        # 测试3：VM碳强度特征
        assert test_vm_carbon_features(), "VM碳强度特征测试失败"
        
        # 测试4：GNN特征映射
        assert test_gnn_features(), "GNN特征映射测试失败"
        
        # 测试5：碳成本计算
        assert test_carbon_cost_calculation(), "碳成本计算测试失败"
        
        # 总结
        print("=" * 80)
        print("🎉 所有测试通过！碳强度特征已成功集成！")
        print("=" * 80)
        print("\n")
        print("✅ 已完成的功能：")
        print("  1. ✓ 碳强度数据配置（4个Host × 24小时）")
        print("  2. ✓ Host生成逻辑（强制4个Host）")
        print("  3. ✓ VM碳强度特征扩展")
        print("  4. ✓ GNN特征空间集成")
        print("  5. ✓ 碳成本计算接口")
        print("\n")
        print("⏳ 待完成的工作：")
        print("  - 修改奖励函数，添加碳成本组件")
        print("  - 调整多目标奖励权重")
        print("  - 重新训练模型")
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

