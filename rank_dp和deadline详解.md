# rank_dp 和 deadline 详解

## 📌 概述

在预调度阶段，我们为每个任务（Task）计算了两个关键属性：
1. **`rank_dp`** - 任务优先级分数
2. **`deadline`** - 任务的子截止时间

这两个属性源自论文中的 **Deadline Partition (DP)** 算法，用于指导后续的任务调度决策。

---

## 🎯 1. rank_dp（任务优先级分数）

### 定义

`rank_dp` 是一个**数值型优先级分数**，表示任务在工作流中的**关键程度**和**紧迫性**。

### 计算方法

`rank_dp` 采用**自底向上（从出口到入口）的递归计算**：

```python
# 伪代码
对于每个任务 task（按逆拓扑序遍历）:
    if task 是出口任务（没有后继）:
        rank_dp[task] = avg_worktime[task]
    else:
        max_child_rank = max(rank_dp[child] for child in task的后继任务)
        rank_dp[task] = max_child_rank + avg_worktime[task]
```

### 含义解释

**rank_dp 越大 → 任务越关键/紧迫**

`rank_dp` 实际上表示：
- **从当前任务到工作流出口的最长路径长度**（以执行时间计）
- 类似于关键路径法（CPM）中的"最晚开始时间"的倒数

### 举例说明

假设有一个简单的工作流：

```
Task A (100ms) → Task B (50ms) → Task D (30ms)
       ↓
Task C (80ms) ──────────────────→ Task D (30ms)
```

计算过程（从后向前）：
1. **Task D**（出口任务）：
   ```
   rank_dp[D] = avg_worktime[D] = 30
   ```

2. **Task B**：
   ```
   rank_dp[B] = rank_dp[D] + avg_worktime[B]
              = 30 + 50 = 80
   ```

3. **Task C**：
   ```
   rank_dp[C] = rank_dp[D] + avg_worktime[C]
              = 30 + 80 = 110
   ```

4. **Task A**（入口任务）：
   ```
   rank_dp[A] = max(rank_dp[B], rank_dp[C]) + avg_worktime[A]
              = max(80, 110) + 100
              = 110 + 100 = 210
   ```

**结论**：Task A 的 rank_dp 最高（210），因为它在关键路径上且距离出口最远。

### 用途

在 PPO 强化学习调度中，`rank_dp` 可用于：

1. **任务优先级排序**：
   ```python
   # rank_dp 越大的任务越优先调度
   sorted_tasks = sorted(tasks, key=lambda t: t.rank_dp, reverse=True)
   ```

2. **状态特征**：
   ```python
   # 作为神经网络的输入特征
   task_features = [task.rank_dp, task.length, task.req_memory_mb, ...]
   ```

3. **奖励函数设计**：
   ```python
   # 优先调度高 rank_dp 的任务可以获得更多奖励
   reward = -task.rank_dp * delay_time
   ```

---

## ⏰ 2. deadline（任务子截止时间）

### 定义

`deadline` 是为每个任务分配的**子截止时间**，表示该任务应该在什么时间点之前完成。

### 计算方法

基于 `rank_dp` 和工作流的总截止时间 `workflow.deadline` 计算：

```python
# 计算公式
rank_dp_0 = max(rank_dp[入口任务])  # 入口任务的最大 rank_dp

对于每个任务 task:
    task.deadline = workflow.deadline * (rank_dp_0 - rank_dp[task] + avg_worktime[task]) / rank_dp_0
```

### 含义解释

这个公式的设计思想：

1. **比例分配**：根据任务在关键路径上的位置，按比例分配时间
2. **公平性**：确保所有任务的时间分配与其重要性成正比
3. **约束性**：入口任务的 deadline 接近 0，出口任务的 deadline 接近 workflow.deadline

### 举例说明

继续上面的例子，假设 `workflow.deadline = 300ms`：

```
rank_dp_0 = 210（Task A 的 rank_dp）
```

计算各任务的 deadline：

1. **Task A**（入口任务）：
   ```
   deadline[A] = 300 * (210 - 210 + 100) / 210
               = 300 * 100 / 210
               = 142.86 ms
   ```

2. **Task B**：
   ```
   deadline[B] = 300 * (210 - 80 + 50) / 210
               = 300 * 180 / 210
               = 257.14 ms
   ```

3. **Task C**：
   ```
   deadline[C] = 300 * (210 - 110 + 80) / 210
               = 300 * 180 / 210
               = 257.14 ms
   ```

4. **Task D**（出口任务）：
   ```
   deadline[D] = 300 * (210 - 30 + 30) / 210
               = 300 * 210 / 210
               = 300 ms
   ```

**观察**：
- Task A（入口）的 deadline = 142.86ms（最早）
- Task D（出口）的 deadline = 300ms（最晚，等于工作流截止时间）
- Task B 和 C 的 deadline 在中间

### 用途

在 PPO 强化学习调度中，`deadline` 可用于：

1. **硬约束**：
   ```python
   # 检查任务是否超期
   if task.finish_time > task.deadline:
       penalty = -1000  # 超期惩罚
   ```

2. **软约束（奖励函数）**：
   ```python
   # 根据是否在 deadline 内完成给予不同奖励
   tardiness = max(0, task.finish_time - task.deadline)
   reward = -tardiness  # 越接近 deadline 完成，奖励越高
   ```

3. **紧迫性度量**：
   ```python
   # 计算任务的紧迫程度
   urgency = (task.deadline - current_time) / task.avg_worktime
   # urgency 越小表示越紧迫
   ```

4. **决策优先级**：
   ```python
   # 优先调度 deadline 紧迫的任务
   sorted_tasks = sorted(tasks, key=lambda t: t.deadline)
   ```

---

## 🔗 3. rank_dp 和 deadline 的关系

### 关键关系

1. **反向关系**：
   - `rank_dp` 越大 → `deadline` 越小（越早）
   - `rank_dp` 越小 → `deadline` 越大（越晚）

2. **互补性**：
   - `rank_dp` 表示"从前往后看"的重要性
   - `deadline` 表示"从后往前看"的时间约束

3. **协同作用**：
   ```python
   # 综合两个指标进行调度决策
   priority = α * rank_dp + β * (1 / (deadline - current_time))
   ```

### 直观理解

想象一个工作流是一条生产线：

- **rank_dp**：表示这个任务"影响下游的程度"
  - 入口任务影响整条生产线 → rank_dp 最大
  - 出口任务只影响自己 → rank_dp 最小

- **deadline**：表示这个任务"必须完成的时间点"
  - 入口任务必须尽早开始 → deadline 最小
  - 出口任务可以最晚完成 → deadline 最大（=工作流deadline）

---

## 📊 4. 实际测试数据示例

从 `test_pre_scheduling.py` 的输出：

```
任务 0:
   - length: 509
   - avg_est: 0.00
   - avg_eft: 0.25
   - rank_dp: 0.71      ← 优先级分数（关键路径长度）
   - deadline: 0.30     ← 子截止时间
   - parent_ids: []     ← 入口任务
   - child_ids: [1, 2, 3, 4, 5]

任务 1:
   - length: 502
   - avg_est: 0.25
   - avg_eft: 0.50
   - rank_dp: 0.25      ← 较小的 rank_dp（非关键路径）
   - deadline: 0.86     ← 较大的 deadline（更宽松）
   - parent_ids: [0]
   - child_ids: []      ← 出口任务
```

**分析**：
- Task 0 是入口任务：rank_dp = 0.71（高）, deadline = 0.30（紧）
- Task 1 是出口任务：rank_dp = 0.25（低）, deadline = 0.86（松）
- 这符合预期：入口任务更关键但必须更早完成

---

## 🎓 5. 理论来源

这两个概念来自论文中的调度算法：

1. **rank_dp** 类似于 HEFT 算法中的 "upward rank"
   - 表示任务到出口的最长路径
   - 用于确定任务的调度优先级

2. **deadline** 基于截止时间划分（Deadline Partition）
   - 将工作流的总截止时间公平地分配给各个任务
   - 考虑了任务的关键程度和执行时间

---

## 💡 6. 如何在 PPO 中使用

### 方案 1: 作为状态特征

```python
def get_state_features(task):
    return np.array([
        task.rank_dp / max_rank_dp,        # 归一化优先级
        (task.deadline - current_time) / max_time,  # 归一化剩余时间
        task.length / max_length,
        task.req_memory_mb / max_memory,
        # ... 其他特征
    ])
```

### 方案 2: 作为奖励函数

```python
def compute_reward(task, finish_time):
    # 基于 deadline 的惩罚
    tardiness = max(0, finish_time - task.deadline)
    deadline_penalty = -tardiness * 10
    
    # 基于 rank_dp 的权重
    importance_weight = task.rank_dp / max_rank_dp
    
    # 综合奖励
    reward = deadline_penalty * importance_weight
    return reward
```

### 方案 3: 作为启发式先验

```python
def get_action_mask(ready_tasks):
    # 优先选择 rank_dp 高且 deadline 紧迫的任务
    sorted_tasks = sorted(ready_tasks, 
                         key=lambda t: (t.rank_dp, -t.deadline),
                         reverse=True)
    return sorted_tasks[:top_k]  # 只考虑前 k 个候选
```

---

## 📝 总结

| 属性 | 含义 | 计算方式 | 典型值范围 | 用途 |
|-----|------|---------|-----------|------|
| **rank_dp** | 任务优先级分数<br/>（关键路径长度） | 自底向上递归<br/>累加执行时间 | 入口大，出口小 | 任务排序<br/>状态特征<br/>奖励权重 |
| **deadline** | 任务子截止时间 | 基于 rank_dp<br/>按比例分配 | 入口小，出口大<br/>（=workflow.deadline） | 时间约束<br/>紧迫性度量<br/>奖励惩罚 |

**核心思想**：通过这两个属性，为 PPO agent 提供了"哪些任务更重要"和"什么时候必须完成"的信息，从而做出更好的调度决策！

