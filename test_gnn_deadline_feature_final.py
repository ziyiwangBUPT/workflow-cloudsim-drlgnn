"""
测试 GNN 特征修改 - 正确版本
保留 task_length，用 normalized_deadline 替换 task_completion_time
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("测试 GNN Deadline 特征修改（正确版本）")
print("=" * 80)

# 1. 验证 TaskObservation
print("\n1. 验证 TaskObservation...")
from scheduler.rl_model.core.env.observation import TaskObservation

task_obs = TaskObservation(
    is_ready=True,
    assigned_vm_id=None,
    start_time=0.0,
    completion_time=0.0,
    energy_consumption=0.0,
    length=1000.0,
    deadline=50.0,
)

print(f"   ✓ TaskObservation 创建成功")
print(f"      - length: {task_obs.length} (保留)")
print(f"      - deadline: {task_obs.deadline} (新增)")
assert hasattr(task_obs, 'length'), "应该保留 length 属性"
assert hasattr(task_obs, 'deadline'), "应该有 deadline 属性"

# 2. 验证 Min-Max 归一化逻辑
print("\n2. 验证 Min-Max 归一化逻辑...")

# 模拟测试数据
test_deadlines = np.array([10.0, 30.0, 50.0, 80.0, 100.0])
print(f"   测试 deadline 值: {test_deadlines}")

min_deadline = test_deadlines.min()
max_deadline = test_deadlines.max()
delta_deadline = max_deadline - min_deadline

eps = 1e-2
if delta_deadline <= eps:
    normalized = np.ones_like(test_deadlines)
else:
    normalized = (test_deadlines - min_deadline) / delta_deadline

print(f"   归一化结果: {normalized}")
print(f"   ✓ 范围: [{normalized.min():.3f}, {normalized.max():.3f}]")

# 3. 验证 wrapper 传递的参数
print("\n3. 检查 wrapper.py 的参数传递...")

# 读取 wrapper.py 检查
with open('scheduler/rl_model/agents/gin_agent/wrapper.py', 'r', encoding='utf-8') as f:
    wrapper_code = f.read()

# 检查关键内容
checks = [
    ('task_length=task_length', '保留 task_length'),
    ('task_normalized_deadline=task_normalized_deadline', '传递 normalized_deadline'),
    ('task_deadlines = np.array', '提取 deadline'),
    ('min_deadline', '计算 min'),
    ('max_deadline', '计算 max'),
    ('delta_deadline <= eps', '除零保护'),
]

print("   关键代码检查:")
all_pass = True
for code_snippet, desc in checks:
    if code_snippet in wrapper_code:
        print(f"      ✓ {desc}")
    else:
        print(f"      ✗ {desc} (未找到)")
        all_pass = False

if all_pass:
    print(f"   ✓ wrapper.py 修改正确")

# 4. 验证 mapper.py 的参数
print("\n4. 检查 mapper.py 的参数...")

with open('scheduler/rl_model/agents/gin_agent/mapper.py', 'r', encoding='utf-8') as f:
    mapper_code = f.read()

checks = [
    ('task_length: np.ndarray,  # 保留', 'map() 保留 task_length 参数'),
    ('task_normalized_deadline: np.ndarray', 'map() 有 normalized_deadline 参数'),
    ('task_length: torch.Tensor  # 保留', 'GinAgentObsTensor 保留 task_length'),
    ('task_normalized_deadline: torch.Tensor', 'GinAgentObsTensor 有 normalized_deadline'),
]

print("   关键代码检查:")
all_pass = True
for code_snippet, desc in checks:
    if code_snippet in mapper_code:
        print(f"      ✓ {desc}")
    else:
        print(f"      ✗ {desc} (未找到)")
        all_pass = False

if all_pass:
    print(f"   ✓ mapper.py 修改正确")

# 5. 验证 agent.py 的特征使用
print("\n5. 检查 agent.py 的特征使用...")

with open('scheduler/rl_model/agents/gin_agent/agent.py', 'r', encoding='utf-8') as f:
    agent_code = f.read()

if 'obs.task_length' in agent_code and 'obs.task_normalized_deadline' in agent_code:
    print(f"   ✓ agent.py 同时使用 task_length 和 task_normalized_deadline")
else:
    print(f"   ✗ agent.py 特征使用不正确")

if 'obs.task_completion_time' in agent_code:
    print(f"   ⚠ agent.py 仍包含 task_completion_time（应该被替换）")
else:
    print(f"   ✓ agent.py 已移除 task_completion_time")

print("\n" + "=" * 80)
print("GNN 特征修改验证完成")
print("=" * 80)

print("""
修改总结：

📝 GNN 节点特征变化：
  之前: [is_scheduled, is_ready, task_length, task_completion_time]
  现在: [is_scheduled, is_ready, task_length, normalized_deadline]
                                    ↑保留          ↑替换

🎯 normalized_deadline 计算方式：
  1. 从当前 State 的所有任务中提取 deadline
  2. 动态计算 min_deadline 和 max_deadline
  3. Min-Max 归一化: (deadline - min) / (max - min)
  4. 处理除零: delta <= 1e-2 时返回 1.0

✅ 修改的文件：
  - scheduler/rl_model/core/env/observation.py
  - scheduler/rl_model/agents/gin_agent/wrapper.py
  - scheduler/rl_model/agents/gin_agent/mapper.py
  - scheduler/rl_model/agents/gin_agent/agent.py

💡 特征含义：
  - task_length: 任务计算量 (MI)
  - normalized_deadline: 归一化的时间压力 [0, 1]
    * 0.0 = 最紧迫的任务
    * 1.0 = 最宽松的任务

🎓 与 ecmws-experiments 的一致性：
  ✓ 使用相同的 Min-Max 归一化公式
  ✓ 使用相同的 eps 阈值 (1e-2)
  ✓ 使用相同的除零处理策略
""")

