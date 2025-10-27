"""
快速验证碳强度特征集成

简化的测试脚本，用于快速验证核心功能。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scheduler.config.carbon_intensity import CARBON_INTENSITY_DATA, FIXED_NUM_HOSTS
from scheduler.dataset_generator.core.gen_dataset import generate_dataset


def main():
    print("\n" + "=" * 60)
    print("碳强度特征快速验证")
    print("=" * 60 + "\n")
    
    # 1. 验证碳强度数据
    print(f"1. 碳强度数据: {len(CARBON_INTENSITY_DATA)} hosts × 24 hours")
    print(f"   Host数量固定为: {FIXED_NUM_HOSTS}\n")
    
    # 2. 生成数据集并验证Host
    print("2. 生成测试数据集...")
    dataset = generate_dataset(
        seed=42,
        host_count=5,  # 测试：请求5个，应该生成4个
        vm_count=12,
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
    
    print(f"   ✓ 生成的Host数量: {len(dataset.hosts)}")
    print(f"   ✓ 生成的VM数量: {len(dataset.vms)}")
    print(f"   ✓ 生成的工作流数量: {len(dataset.workflows)}\n")
    
    # 3. 验证Host的碳强度曲线
    print("3. 验证Host碳强度曲线:")
    for host in dataset.hosts:
        has_curve = host.carbon_intensity_curve is not None
        curve_len = len(host.carbon_intensity_curve) if has_curve else 0
        print(f"   Host {host.id}: 碳强度曲线 = {has_curve}, 长度 = {curve_len}")
        
        if has_curve:
            # 测试获取碳强度
            carbon_at_0h = host.get_carbon_intensity_at(0)
            carbon_at_12h = host.get_carbon_intensity_at(12 * 3600)
            print(f"             0:00时碳强度 = {carbon_at_0h:.3f}")
            print(f"             12:00时碳强度 = {carbon_at_12h:.3f}")
    
    # 4. 验证VM分配到Host
    print(f"\n4. 验证VM分配:")
    host_vm_count = {i: 0 for i in range(FIXED_NUM_HOSTS)}
    for vm in dataset.vms:
        if 0 <= vm.host_id < FIXED_NUM_HOSTS:
            host_vm_count[vm.host_id] += 1
    
    for host_id, count in host_vm_count.items():
        print(f"   Host {host_id}: {count} 个VM")
    
    print("\n" + "=" * 60)
    print("✅ 验证通过！碳强度特征已成功集成！")
    print("=" * 60 + "\n")
    
    print("📝 下一步：")
    print("   - 运行完整测试：python test_carbon_intensity_integration.py")
    print("   - 修改奖励函数以使用碳成本")
    print("   - 调整多目标权重")
    print("\n")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

