"""最小测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scheduler.dataset_generator.core.gen_dataset import generate_dataset
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment

# 生成小数据集
dataset = generate_dataset(
    seed=42,
    host_count=4,
    vm_count=10,
    workflow_count=2,  # 只生成2个工作流
    gnp_min_n=5,
    gnp_max_n=5,  # 每个工作流5个任务
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

# 重置环境
obs, info = env.reset()

print("✓ 环境创建和重置成功！")
print(f"✓ 状态: {env.state is not None}")
print(f"✓ 时钟管理器: {env.state.clock_manager is not None}")

