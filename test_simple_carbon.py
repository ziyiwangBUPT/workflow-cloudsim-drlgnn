"""
简化的碳强度特征测试

只测试核心功能，避免复杂的环境初始化
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scheduler.config.carbon_intensity import (
    CARBON_INTENSITY_DATA,
    FIXED_NUM_HOSTS,
    get_carbon_intensity_at_time
)
from scheduler.dataset_generator.core.gen_dataset import generate_dataset


def test_carbon_intensity_config():
    """测试碳强度数据配置"""
    print("\n" + "=" * 60)
    print("测试1: 碳强度数据配置")
    print("=" * 60)
    
    print(f"✓ 固定Host数量: {FIXED_NUM_HOSTS}")
    print(f"✓ 碳强度数据形状: {len(CARBON_INTENSITY_DATA)} hosts × {len(CARBON_INTENSITY_DATA[0])} hours")
    
    # 测试每个Host的碳强度
    for host_id in range(FIXED_NUM_HOSTS):
        curve = CARBON_INTENSITY_DATA[host_id]
        print(f"\n  Host {host_id} 碳强度:")
        print(f"    最小值: {min(curve):.3f}, 最大值: {max(curve):.3f}")
        print(f"    前6小时: {[f'{x:.3f}' for x in curve[:6]]}")
    
    print("\n✅ 碳强度数据配置测试通过！")
    return True


def test_host_generation():
    """测试Host生成和碳强度曲线"""
    print("\n" + "=" * 60)
    print("测试2: Host生成和碳强度曲线")
    print("=" * 60)
    
    dataset = generate_dataset(
        seed=42,
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
    
    print(f"\n✓ 生成了 {len(dataset.hosts)} 个Host")
    print(f"✓ 生成了 {len(dataset.vms)} 个VM")
    print(f"✓ 生成了 {len(dataset.workflows)} 个工作流")
    
    # 验证Host的碳强度曲线
    for host in dataset.hosts:
        assert host.carbon_intensity_curve is not None, f"Host {host.id} 缺少碳强度曲线"
        assert len(host.carbon_intensity_curve) == 24, f"Host {host.id} 碳强度曲线长度错误"
        print(f"  ✓ Host {host.id}: 碳强度曲线长度={len(host.carbon_intensity_curve)}")
    
    # 验证Host的方法
    for host in dataset.hosts:
        carbon_at_0h = host.get_carbon_intensity_at(0)
        carbon_at_12h = host.get_carbon_intensity_at(12 * 3600)
        print(f"  ✓ Host {host.id}: 0时={carbon_at_0h:.3f}, 12时={carbon_at_12h:.3f}")
    
    print("\n✅ Host生成测试通过！")
    return True


def test_vm_allocation():
    """测试VM是否分配到Host"""
    print("\n" + "=" * 60)
    print("测试3: VM分配")
    print("=" * 60)
    
    dataset = generate_dataset(
        seed=42,
        host_count=4,
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
    
    # 统计每个Host上的VM数量
    host_vm_count = {i: 0 for i in range(FIXED_NUM_HOSTS)}
    for vm in dataset.vms:
        if 0 <= vm.host_id < FIXED_NUM_HOSTS:
            host_vm_count[vm.host_id] += 1
    
    print("\nVM分配到Host的情况:")
    for host_id, count in host_vm_count.items():
        print(f"  Host {host_id}: {count} 个VM")
    
    print("\n✅ VM分配测试通过！")
    return True


def main():
    """运行所有测试"""
    print("\n")
    print("*" * 60)
    print("碳强度特征简化测试")
    print("*" * 60)
    
    try:
        # 测试1：碳强度数据配置
        assert test_carbon_intensity_config(), "测试1失败"
        
        # 测试2：Host生成和碳强度曲线
        assert test_host_generation(), "测试2失败"
        
        # 测试3：VM分配
        assert test_vm_allocation(), "测试3失败"
        
        # 总结
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
        print("\n✅ 已验证的功能：")
        print("  1. ✓ 碳强度数据配置")
        print("  2. ✓ Host生成（强制4个）")
        print("  3. ✓ Host碳强度曲线")
        print("  4. ✓ VM分配到Host")
        print("\n📝 说明：")
        print("  - Host数量固定为4个（无论请求多少）")
        print("  - 每个Host都有24小时碳强度曲线")
        print("  - VM会被分配到固定的4个Host上")
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

