# 如何使用 Pegasus 真实工作流模板

## 🚀 快速开始

### 1. 最简单的使用方式

只需要将 `dag_method` 参数改为 `'pegasus'`：

```python
from scheduler.dataset_generator.core.gen_dataset import generate_dataset

dataset = generate_dataset(
    seed=42,
    host_count=5,
    vm_count=15,
    workflow_count=3,
    gnp_min_n=5,              # 使用 pegasus 时这个参数不起作用
    gnp_max_n=10,             # 使用 pegasus 时这个参数不起作用
    max_memory_gb=16,
    min_cpu_speed_mips=1000,
    max_cpu_speed_mips=3000,
    dag_method='pegasus',     # ← 改这里！从 'gnp' 改为 'pegasus'
    task_length_dist='uniform',
    min_task_length=10000,
    max_task_length=100000,
    task_arrival='static',
    arrival_rate=1.0
)
```

就这么简单！✨

---

## 📊 测试结果

运行测试脚本：
```bash
cd paper1115
python test_pegasus_workflows.py
```

您将看到：
- ✅ 3 个真实工作流（总共 105 个任务）
- ✅ 复杂的 DAG 结构（不再是简单的星形）
- ✅ 每个任务都有正确的 `rank_dp` 和 `deadline`
- ✅ 工作流结构分析和统计

---

## 🎯 对比：GNP vs Pegasus

### 之前（GNP - 问题结构）
```
工作流 0:
  - 任务数: 8
  - 入口任务: 1 个
  - 中间任务: 0 个    ← 问题！没有中间层
  - 出口任务: 7 个     ← 所有任务都直接连到出口

结构示意：
    0 (入口)
   /||\\\
  1 2 3 4 5 6 7  (都是出口)
```

### 现在（Pegasus - 真实结构）
```
工作流 0:
  - 任务数: 33
  - 入口任务: 1 个
  - 中间任务: 31 个   ← 有层次结构！
  - 出口任务: 1 个

结构示意（简化）：
    0 (入口)
    ├─→ 1 ─→ 24 ─→ 32 (出口)
    ├─→ 2 ─→ 24 ─┘
    ├─→ 3 ─→ 24 ─┘
    └─→ ... (更多路径)
```

---

## 📁 可用的工作流模板

项目包含 6 个真实科学工作流（自动随机选择）：

1. **example.dag** - Diamond 结构（简单，适合测试）
2. **montage.dag** - 天文图像拼接（大规模并行）
3. **genome.dag** - 基因组分析（流水线结构）
4. **sipht.dag** - RNA 分析（中等复杂度）
5. **cybershake.dag** - 地震模拟（高度并行）
6. **inspiral.dag** - 引力波分析（分层结构）

---

## 💡 在 gym_env.py 中使用

预调度逻辑已经支持任何 DAG 结构，无需修改！

只需在创建环境时指定参数：

```python
from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment

dataset_args = DatasetArgs(
    host_count=5,
    vm_count=15,
    workflow_count=3,
    gnp_min_n=5,
    gnp_max_n=10,
    dag_method='pegasus',  # ← 使用 Pegasus
    # ... 其他参数
)

env = CloudSchedulingGymEnvironment(dataset_args=dataset_args)
state, info = env.reset()

# WS 和 DP 预调度已自动完成！
# 所有任务都有 rank_dp 和 deadline
```

---

## 🎓 实际测试数据

从 `test_pegasus_workflows.py` 的输出：

```
工作流 0 (Montage):
   - 任务数: 33
   - avg_eft: 179.55 秒
   - deadline: 215.47 秒
   - rank_dp 范围: 10.41 ~ 179.55
   - deadline 范围: 16.60 ~ 215.47

示例任务：
   Task 0 (入口):
      length: 27143 MI
      rank_dp: 179.55 (最高优先级)
      deadline: 16.60 秒 (最早截止)
      children: [1, 2, 3, ..., 31] (31个后继任务)
   
   Task 1 (中间):
      length: 84160 MI
      rank_dp: 163.03
      deadline: 71.30 秒
      parents: [0], children: [24]
   
   Task 32 (出口):
      length: 20416 MI
      rank_dp: 10.41 (最低优先级)
      deadline: 215.47 秒 (最晚截止)
```

---

## ✅ 验证清单

运行 `test_pegasus_workflows.py` 后检查：

- ✅ 工作流有合理的层次结构（入口、中间、出口任务）
- ✅ 每个任务的 `rank_dp > 0`
- ✅ 每个任务的 `deadline > 0`
- ✅ 入口任务的 `rank_dp` 最大，`deadline` 最小
- ✅ 出口任务的 `rank_dp` 最小，`deadline` 最大

---

## 🔧 故障排除

### 问题：还是看到星形结构？

确保您使用了 `dag_method='pegasus'` 而不是 `dag_method='gnp'`。

### 问题：任务数量固定？

是的！Pegasus 模板的任务数量是固定的（由 .dag 文件决定）。不同的模板有不同的任务数量。

### 问题：想要特定的工作流？

临时修改 `scheduler/config/settings.py`，只保留想要的模板：

```python
WORKFLOW_FILES = [
    DATA_PATH / "pegasus_workflows" / "example.dag",  # 只使用 example
]
```

---

## 📚 更多信息

详细说明请查看：`Pegasus工作流使用说明.md`

---

**总结**：只需将 `dag_method='gnp'` 改为 `dag_method='pegasus'`，立即获得真实的工作流结构！🎉

