# Pegasus 真实工作流模板使用说明

## 📋 概述

Pegasus 是一个著名的科学工作流管理系统。本项目已集成了 6 个真实的 Pegasus 工作流模板，可以生成更符合实际场景的任务调度测试数据。

---

## 🗂️ 可用的工作流模板

项目中包含以下 6 个真实工作流模板（位于 `data/pegasus_workflows/`）：

| 工作流名称 | 文件 | 来源领域 | 特点 |
|-----------|------|---------|------|
| **Example** | `example.dag` | 示例 | Diamond 结构，简单易懂 |
| **Montage** | `montage.dag` | 天文学 | 图像拼接，复杂并行 |
| **Genome** | `genome.dag` | 基因组学 | 基因序列分析，流水线结构 |
| **SIPHT** | `sipht.dag` | 生物信息学 | RNA 分析，中等复杂度 |
| **CyberShake** | `cybershake.dag` | 地震学 | 地震波模拟，大规模并行 |
| **Inspiral** | `inspiral.dag` | 物理学 | 引力波分析，分层结构 |

---

## 🚀 使用方法

### 方法 1：在代码中使用

```python
from scheduler.dataset_generator.core.gen_dataset import generate_dataset

# 生成使用 Pegasus 工作流的数据集
dataset = generate_dataset(
    seed=42,
    host_count=5,
    vm_count=15,
    workflow_count=3,
    gnp_min_n=5,              # pegasus 模式下此参数不使用
    gnp_max_n=10,             # pegasus 模式下此参数不使用
    max_memory_gb=16,
    min_cpu_speed_mips=1000,
    max_cpu_speed_mips=3000,
    dag_method='pegasus',     # ← 关键参数：使用 pegasus
    task_length_dist='uniform',
    min_task_length=10000,
    max_task_length=100000,
    task_arrival='static',
    arrival_rate=1.0
)

# 每次运行会随机从 6 个模板中选择工作流
print(f"生成了 {len(dataset.workflows)} 个工作流")
```

### 方法 2：运行测试脚本

我已经为您创建了专门的测试脚本：

```bash
cd paper1115
python test_pegasus_workflows.py
```

这个脚本会：
1. 生成使用 Pegasus 模板的工作流
2. 分析工作流结构
3. 运行 WS 和 DP 预调度算法
4. 显示详细的任务信息和统计

---

## 📊 Pegasus vs GNP 对比

### GNP 模型（随机图）
```python
dag_method='gnp'
```
- **优点**：灵活，可控制任务数量
- **缺点**：结构随机，可能不真实
- **典型结构**：星形或随机连接
- **适用场景**：算法开发、快速测试

### Pegasus 模板（真实工作流）
```python
dag_method='pegasus'
```
- **优点**：真实场景，复杂结构
- **缺点**：任务数量固定（由模板决定）
- **典型结构**：流水线、分层、并行
- **适用场景**：性能评估、论文实验

---

## 🔍 工作流结构示例

### Example (Diamond) 工作流
```
            [preprocess]
               /    \
    [findrange1]  [findrange2]
               \    /
             [analyze]
```

特点：
- 入口任务：1 个
- 中间任务：2 个（并行）
- 出口任务：1 个
- 适合：测试并行调度

### Montage 工作流
```
[mProject_0] → [mDiffFit_0] → [mConcatFit] → [mBgModel] → [mBackground_0] → [mAdd]
[mProject_1] → [mDiffFit_1] ↗                                              ↗
[mProject_2] → [mDiffFit_2] ↗                             [mBackground_1] ↗
```

特点：
- 多层流水线结构
- 高度并行化
- 适合：测试复杂调度策略

---

## 📈 预期输出

运行 `test_pegasus_workflows.py` 后，您将看到：

```
================================================================================
使用 Pegasus 真实工作流模板测试
================================================================================

1. 生成数据集（使用 Pegasus 工作流）...
   - 生成了 3 个工作流
   - 生成了 15 个虚拟机
   - 总任务数: 45

2. 工作流结构分析...

   工作流 0:
   - 任务数: 15
   - 入口任务: 1 个
   - 中间任务: 8 个
   - 出口任务: 6 个
   - 平均分支因子: 1.53
   - 任务结构示例（前3个）:
      Task 0: parents=[], children=[1, 2, 3]
      Task 1: parents=[0], children=[4]
      Task 2: parents=[0], children=[5, 6]

3. 预计算工作流数据...
   - 工作流 0: avg_eft=78.45, workload=45230.00, deadline=94.14
   - 工作流 1: avg_eft=82.31, workload=48120.00, deadline=98.77
   - 工作流 2: avg_eft=75.89, workload=42890.00, deadline=91.07

4. 运行工作流排序 (WS) 算法...
   - 工作流已排序，顺序: [2, 0, 1]

5. 运行截止时间划分 (DP) 算法...

6. 验证每个任务的属性...
   ✓ 所有任务都已正确设置 rank_dp 和 deadline 属性

7. 第一个工作流的详细任务信息:

   入口任务 (1 个):
      Task 0:
         length: 28809, rank_dp: 78.45, deadline: 15.32
         children: [1, 2, 3, 4]

   中间任务 (8 个, 显示前3个):
      Task 1:
         length: 15234, rank_dp: 52.38, deadline: 45.67
         parents: [0], children: [5, 6]
      Task 2:
         length: 18945, rank_dp: 48.92, deadline: 52.34
         parents: [0], children: [7]
      ...

   出口任务 (6 个, 显示前2个):
      Task 10:
         length: 12345, rank_dp: 12.34, deadline: 89.23
         parents: [5, 6]
      ...

8. rank_dp 分布分析:
   - 最小 rank_dp: 12.34
   - 最大 rank_dp: 78.45
   - 平均 rank_dp: 42.56

9. deadline 分布分析:
   工作流 0:
      - deadline 范围: 15.32 ~ 92.45
      - 工作流 deadline: 94.14

================================================================================
Pegasus 工作流测试完成！
================================================================================

✓ 所有测试通过！Pegasus 工作流已成功集成。
```

---

## ⚙️ 配置说明

### 修改工作流模板列表

编辑 `scheduler/config/settings.py`：

```python
WORKFLOW_FILES = [
    DATA_PATH / "pegasus_workflows" / "example.dag",
    DATA_PATH / "pegasus_workflows" / "montage.dag",
    # 添加或删除工作流模板
]
```

### 工作流选择机制

当使用 `dag_method='pegasus'` 时：
- 系统会从 `WORKFLOW_FILES` 列表中**随机选择**一个工作流模板
- 每次调用 `generate_dataset()` 可能生成不同的工作流类型
- 如果 `workflow_count=3`，则会生成 3 个随机选择的工作流

---

## 🔧 高级用法

### 1. 指定使用特定工作流（需要修改代码）

如果您想只使用特定的工作流，可以临时修改 `settings.py`：

```python
# 只使用 Diamond 工作流
WORKFLOW_FILES = [
    DATA_PATH / "pegasus_workflows" / "example.dag",
]
```

### 2. 混合使用 GNP 和 Pegasus

您可以为不同的实验使用不同的方法：

```python
# 开发阶段：使用 GNP（快速测试）
dev_dataset = generate_dataset(
    seed=1,
    dag_method='gnp',
    gnp_min_n=5,
    gnp_max_n=10,
    # ...
)

# 评估阶段：使用 Pegasus（真实场景）
eval_dataset = generate_dataset(
    seed=2,
    dag_method='pegasus',
    # ...
)
```

### 3. 在 gym_env.py 中使用

`gym_env.py` 的预调度逻辑已经支持任何 DAG 结构，无需修改。只需在创建环境时指定 `dag_method='pegasus'`：

```python
from scheduler.dataset_generator.gen_dataset import DatasetArgs

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
```

---

## 📝 总结

| 方面 | GNP 模式 | Pegasus 模式 |
|-----|---------|-------------|
| 结构 | 随机生成 | 真实模板 |
| 任务数 | 可控 (gnp_min_n ~ gnp_max_n) | 固定（由模板决定） |
| 复杂度 | 简单到中等 | 真实复杂 |
| 多样性 | 每次都不同 | 6 种模板随机选择 |
| 推荐用途 | 开发、调试 | 评估、发表 |

**建议工作流程**：
1. 🔧 **开发阶段**：使用 `dag_method='gnp'` 快速测试
2. 🧪 **调试阶段**：使用 `dag_method='pegasus'` + `example.dag` 验证
3. 📊 **评估阶段**：使用 `dag_method='pegasus'` + 所有模板进行性能测试
4. 📄 **论文实验**：使用 `dag_method='pegasus'` 确保结果可信度

---

## 🎉 快速开始

```bash
# 1. 运行 Pegasus 测试
cd paper1115
python test_pegasus_workflows.py

# 2. 查看工作流结构
# 观察输出中的工作流结构分析部分

# 3. 在您的代码中使用
# 将 dag_method='gnp' 改为 dag_method='pegasus'
```

祝您实验顺利！🚀

