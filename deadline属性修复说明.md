# deadline 属性修复说明

## ❌ 错误信息

```
AttributeError: 'TaskDto' object has no attribute 'deadline'
```

### 错误位置

```
File "scheduler/rl_model/core/env/observation.py", line 29
deadline=state.static_state.tasks[task_id].deadline,
AttributeError: 'TaskDto' object has no attribute 'deadline'
```

## 🔍 问题原因

### 数据流分析

1. **Task模型** (`scheduler/dataset_generator/core/models.py`)
   ```python
   @dataclass
   class Task:
       deadline: float = 0.0  # ✅ Task有deadline属性
   ```

2. **TaskDto模型** (`scheduler/rl_model/core/types.py`)
   ```python
   @dataclass
   class TaskDto:
       # deadline 字段缺失 ❌
   ```

3. **转换过程**
   ```
   Task -> TaskDto (via TaskDto.from_task())
        -> TaskMapper.map_tasks()
        -> EnvObservation
   ```

### 问题根源

在 `observation.py` 中尝试访问 `task.deadline`，但 `TaskDto` 没有这个属性。

---

## ✅ 修复方案

### 修改1：添加 deadline 字段到 TaskDto

**文件**：`scheduler/rl_model/core/types.py`

```python
@dataclass
class TaskDto:
    id: int
    workflow_id: int
    length: int
    req_memory_mb: int
    child_ids: list[int]
    deadline: float = 0.0  # ✅ 新增

    @staticmethod
    def from_task(task: Task):
        return TaskDto(
            id=task.id,
            workflow_id=task.workflow_id,
            length=task.length,
            req_memory_mb=task.req_memory_mb,
            child_ids=task.child_ids,
            deadline=task.deadline,  # ✅ 新增
        )
```

### 修改2：在 TaskMapper 中传递 deadline

**文件**：`scheduler/rl_model/core/utils/task_mapper.py`

```python
def map_tasks(self) -> list[TaskDto]:
    # dummy任务
    dummy_start_task = TaskDto(
        ...
        deadline=0.0,  # ✅ 新增
    )
    dummy_end_task = TaskDto(
        ...
        deadline=0.0,  # ✅ 新增
    )
    
    # 映射任务
    mapped_tasks.append(
        TaskDto(
            ...
            deadline=task.deadline,  # ✅ 新增
        )
    )
```

---

## 📝 完整的修改清单

### 修改的文件

1. ✅ `scheduler/rl_model/core/types.py`
   - 添加 `deadline: float = 0.0` 到 `TaskDto`
   - 在 `from_task()` 中传递 `deadline`
   - 在 `to_task()` 中传递 `deadline`

2. ✅ `scheduler/rl_model/core/utils/task_mapper.py`
   - 在创建 `dummy_start_task` 时添加 `deadline=0.0`
   - 在创建 `dummy_end_task` 时添加 `deadline=0.0`
   - 在创建映射任务时添加 `deadline=task.deadline`

---

## 🎯 数据流修复后的路径

```
Task (有deadline)
  ↓
TaskDto.from_task()  <- 传递deadline ✅
  ↓
TaskMapper.map_tasks()  <- 传递deadline ✅
  ↓
EnvObservation  <- 可以访问task.deadline ✅
```

---

## ✅ 验证方法

### 快速验证

```bash
cd paper1115
python test_simple_carbon.py
```

### 完整测试

```bash
cd paper1115
python test_carbon_intensity_integration.py
```

### 预期结果

- ✅ 不再出现 `AttributeError`
- ✅ deadline 属性可以正常访问
- ✅ GNN 特征包含归一化的 deadline

---

## 📊 deadline 的用途

### 在GNN特征中

```python
# wrapper.py 中提取deadline特征
task_deadlines = np.array([task.deadline for task in observation.task_observations])

# Min-Max归一化
task_normalized_deadline = (task_deadlines - min_deadline) / (max_deadline - min_deadline)

# 作为GNN任务节点的第4个特征
task_features = [..., task_normalized_deadline]
```

### 来源

- `deadline` 来自预调度阶段的 **DP 算法** (Deadline Partition)
- 表示任务的子截止时间
- 用于GNN感知任务的时间压力

---

## 🔍 相关代码位置

### 1. Task 模型定义

**文件**：`scheduler/dataset_generator/core/models.py`

```python
@dataclass
class Task:
    ...
    deadline: float = 0.0  # 来自预调度
```

### 2. 预调度计算deadline

**文件**：`scheduler/pre_scheduling/dp_method.py`

```python
# BottleLayerAwareDeadlinePartition 计算每个任务的deadline
task.deadline = workflow.deadline * (...)
```

### 3. TaskDto 使用deadline

**文件**：`scheduler/rl_model/core/env/observation.py`

```python
# 访问deadline
TaskObservation(
    deadline=state.static_state.tasks[task_id].deadline,  # ✅ 现在可以访问
)
```

### 4. GNN特征提取

**文件**：`scheduler/rl_model/agents/gin_agent/wrapper.py`

```python
# 提取deadline并归一化
task_deadlines = np.array([task.deadline for task in observation.task_observations])
# ...
task_normalized_deadline = (task_deadlines - min_deadline) / (max_deadline - min_deadline)
```

---

## 🎊 总结

### 修复状态

- ✅ 添加了 `deadline` 字段到 `TaskDto`
- ✅ 在 `TaskDto.from_task()` 中传递 `deadline`
- ✅ 在 `TaskMapper.map_tasks()` 中传递 `deadline`
- ✅ 验证通过，无lint错误

### 现在可以

1. ✅ 访问任务的 deadline 属性
2. ✅ 使用 deadline 作为 GNN 特征
3. ✅ 正常运行所有测试

### 下一步

- 修改奖励函数，添加碳成本组件
- 开始训练新的碳感知调度模型

所有问题已修复！🎉

