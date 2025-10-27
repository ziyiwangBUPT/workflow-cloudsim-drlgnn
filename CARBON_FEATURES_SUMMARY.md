# 碳强度特征集成实现总结

## 🎯 实现目标

将 paper1115 项目的调度目标从最小化"能耗"扩展到最小化"碳成本"，模仿 `ecmws-experiments` 项目处理电价的方式。

**核心公式**：
```
碳成本 = 能耗 × 碳强度
```

---

## ✅ 已完成的修改

### 1. 碳强度数据配置模块

**新增文件**：`scheduler/config/carbon_intensity.py`

**功能**：
- 硬编码4个Host的24小时碳强度曲线
- 提供碳强度查询和计算接口

**关键函数**：
```python
# 全局数据
CARBON_INTENSITY_DATA: list[list[float]]  # 4×24 碳强度数组
FIXED_NUM_HOSTS: int = 4

# 接口函数
get_carbon_intensity_at_time(host_id, time_seconds) -> float
get_carbon_intensity_features(host_id, start_time, end_time) -> list[float]
calculate_carbon_cost(energy, host_id, start_time, end_time) -> float
```

---

### 2. Host数据模型扩展

**修改文件**：`scheduler/dataset_generator/core/models.py`

**修改内容**：
```python
@dataclass
class Host:
    # 原有字段...
    carbon_intensity_curve: list[float] = None  # 新增：24小时碳强度曲线
    
    def get_carbon_intensity_at(self, time_seconds: float) -> float:
        """获取指定时间的碳强度值"""
```

---

### 3. Host生成逻辑修改

**修改文件**：`scheduler/dataset_generator/core/gen_vm.py`

**修改内容**：
- 强制生成固定数量（4个）的Host
- 自动为每个Host分配碳强度曲线
- 忽略 `host_count` 参数，始终生成4个Host

**关键代码**：
```python
def generate_hosts(n: int, rng: np.random.RandomState) -> list[Host]:
    actual_n = FIXED_NUM_HOSTS  # 强制为4
    # ...
    carbon_intensity_curve=CARBON_INTENSITY_DATA[i]  # 分配碳强度曲线
```

---

### 4. 虚拟时钟管理系统

**新增文件**：`scheduler/rl_model/core/env/clock_manager.py`

**功能**：
- 为每个工作流维护独立的虚拟时钟
- 支持时钟推进和查询
- 提供任务级别的时钟访问

**关键类**：
```python
class ClockManager:
    def initialize(self, workflows) -> None
    def get_workflow_clock(self, workflow_id) -> float
    def advance_workflow_clock(self, workflow_id, time_delta) -> None
    def update_clock_for_task_completion(self, task_id, completion_time) -> None
```

**Workflow扩展**：
```python
@dataclass
class Workflow:
    virtual_clock: float = 0.0  # 新增
    
    def advance_clock(self, time_delta: float) -> None
    def get_current_hour(self) -> int
```

---

### 5. VM数据模型扩展

**修改文件**：
- `scheduler/rl_model/core/types.py`
- `scheduler/rl_model/core/env/observation.py`

**VmDto扩展**：
```python
@dataclass
class VmDto:
    # 原有字段...
    host_id: int  # 新增
    host_carbon_intensity_curve: list[float]  # 新增
    
    def get_carbon_intensity_at(self, time_seconds: float) -> float
```

**VmObservation扩展**：
```python
@dataclass
class VmObservation:
    # 原有字段...
    host_id: int = 0  # 新增
    host_carbon_intensity_curve: list[float] = None  # 新增
    
    def get_carbon_intensity_at(self, time_seconds: float) -> float
```

---

### 6. GNN特征空间修改

**修改文件**：
- `scheduler/rl_model/agents/gin_agent/agent.py`
- `scheduler/rl_model/agents/gin_agent/mapper.py`
- `scheduler/rl_model/agents/gin_agent/wrapper.py`

**VM节点特征扩展**：
```python
# 原有特征（3维）
vm_features = [vm_completion_time, 1/(vm_speed+1e-8), vm_energy_rate]

# 新特征（4维）
vm_features = [vm_completion_time, 1/(vm_speed+1e-8), vm_energy_rate, vm_carbon_intensity]
```

**GinAgentObsTensor扩展**：
```python
@dataclass
class GinAgentObsTensor:
    # 原有字段...
    vm_carbon_intensity: torch.Tensor  # 新增
```

**mapper扩展**：
- `map()` 方法添加 `vm_carbon_intensity` 参数
- `unmap()` 方法提取 `vm_carbon_intensity`

**wrapper扩展**：
```python
# 在 map_observation() 中提取碳强度特征
vm_carbon_intensity = np.array([
    vm.get_carbon_intensity_at(vm.completion_time) 
    for vm in observation.vm_observations
])
```

---

### 7. 碳成本计算接口

**修改文件**：`scheduler/rl_model/core/env/observation.py`

**新增方法**：
```python
class EnvObservation:
    def carbon_cost(self) -> float:
        """
        计算总碳成本
        碳成本 = ∑(任务能耗 × 碳强度)
        
        为奖励函数预留的接口
        """
```

---

## 🔌 预留的奖励函数接口

### 当前奖励函数

**位置**：`scheduler/rl_model/agents/gin_agent/wrapper.py`

```python
def step(self, action: int):
    # ...
    makespan_reward = -(obs.makespan() - self.prev_obs.makespan()) / obs.makespan()
    energy_reward = -(obs.energy_consumption() - self.prev_obs.energy_consumption()) / obs.energy_consumption()
    reward = makespan_reward + energy_reward  # 当前只有两项
    # ...
```

### 如何添加碳成本（示例）

```python
def step(self, action: int):
    # ...
    makespan_reward = -(obs.makespan() - self.prev_obs.makespan()) / obs.makespan()
    energy_reward = -(obs.energy_consumption() - self.prev_obs.energy_consumption()) / obs.energy_consumption()
    
    # 新增：碳成本奖励
    carbon_cost = obs.carbon_cost()
    prev_carbon_cost = self.prev_obs.carbon_cost()
    carbon_reward = -(carbon_cost - prev_carbon_cost) / max(carbon_cost, 1e-8)
    
    # 多目标奖励（权重可调）
    w1, w2, w3 = 0.33, 0.33, 0.34  # makespan, energy, carbon
    reward = w1 * makespan_reward + w2 * energy_reward + w3 * carbon_reward
    # ...
```

---

## 🧪 测试方式

### 快速验证

```bash
cd paper1115
python verify_carbon_features.py
```

### 完整测试

```bash
cd paper1115
python test_carbon_intensity_integration.py
```

---

## 📊 数据结构示意

### 碳强度数据（4 Hosts × 24 Hours）

```
CARBON_INTENSITY_DATA = [
    [0.15, 0.10, ..., 0.14],  # Host 0: 高碳区域（煤电）
    [0.07, 0.08, ..., 0.09],  # Host 1: 低碳区域（水电）
    [0.10, 0.09, ..., 0.12],  # Host 2: 中碳区域（混合）
    [0.13, 0.13, ..., 0.13],  # Host 3: 高碳区域（天然气）
]
```

### 虚拟时钟流程

```
时间轴（假设工作流入口为0:00）:
|---------|---------|---------|---------|
0:00     3:00     6:00     9:00     12:00
  ↑       ↑        ↑        ↑        ↑
 Task1  Task2    Task3    Task4    ...
  
每个任务调度时:
1. 计算任务开始时间（相对工作流开始）
2. 转换为小时：hour = int(start_time // 3600) % 24
3. 获取该小时的碳强度
4. 计算碳成本 = 能耗 × 碳强度
```

---

## 🎨 架构设计

### 批处理 + 虚拟时钟

```
┌─────────────────────────────────────────┐
│  数据生成阶段                            │
│  - 生成4个固定Host                       │
│  - 为每个Host分配碳强度曲线               │
│  - VM分配到Host                         │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  预调度阶段                              │
│  - 计算工作流deadline                    │
│  - 工作流排序(WS)                        │
│  - 任务deadline划分(DP)                  │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  强化学习调度阶段                        │
│  - 虚拟时钟初始化（所有工作流从0:00开始）│
│  - 逐个任务调度                          │
│  - GNN特征包含碳强度                     │
│  - 根据任务完成时间推进时钟               │
└─────────────────────────────────────────┘
```

---

## 🔧 关键设计决策

### 1. 为什么强制4个Host？
- 保持与ecmws一致的碳强度数据结构
- 简化碳强度管理
- 便于对比实验

### 2. 为什么使用虚拟时钟？
- paper1115是批处理框架，没有全局时钟
- 虚拟时钟避免大规模架构修改
- 每个工作流独立时钟，互不影响

### 3. 碳强度如何集成到GNN？
- 作为VM节点的第4个特征
- 动态计算：根据VM当前完成时间获取碳强度
- 随着调度推进，碳强度特征动态变化

### 4. 为什么不直接修改奖励函数？
- 奖励函数涉及多个组件
- 需要仔细调整权重
- 建议先验证特征集成正确，再调整奖励

---

## 📝 修改文件清单

| 文件 | 类型 | 修改内容 |
|------|------|---------|
| `scheduler/config/carbon_intensity.py` | 新增 | 碳强度数据和工具函数 |
| `scheduler/dataset_generator/core/models.py` | 修改 | Host和Workflow添加碳强度相关字段 |
| `scheduler/dataset_generator/core/gen_vm.py` | 修改 | Host生成逻辑强制为4个 |
| `scheduler/rl_model/core/env/clock_manager.py` | 新增 | 虚拟时钟管理器 |
| `scheduler/rl_model/core/types.py` | 修改 | VmDto添加碳强度字段 |
| `scheduler/rl_model/core/env/observation.py` | 修改 | VmObservation添加碳强度，新增carbon_cost() |
| `scheduler/rl_model/agents/gin_agent/mapper.py` | 修改 | map/unmap添加碳强度特征 |
| `scheduler/rl_model/agents/gin_agent/agent.py` | 修改 | GNN VM特征添加碳强度维度 |
| `scheduler/rl_model/agents/gin_agent/wrapper.py` | 修改 | 提取碳强度特征 |
| `test_carbon_intensity_integration.py` | 新增 | 完整测试脚本 |
| `verify_carbon_features.py` | 新增 | 快速验证脚本 |
| `碳强度特征集成说明.md` | 新增 | 使用说明文档 |

---

## 🚀 快速开始

### 1. 验证安装

```bash
cd paper1115
python verify_carbon_features.py
```

### 2. 运行完整测试

```bash
python test_carbon_intensity_integration.py
```

### 3. 查看碳强度数据

```python
from scheduler.config.carbon_intensity import CARBON_INTENSITY_DATA
print(CARBON_INTENSITY_DATA)
```

---

## 🔄 与ecmws项目的对比

| 特性 | ecmws | paper1115 | 备注 |
|------|-------|-----------|------|
| **数据结构** | 4×24电价数组 | 4×24碳强度数组 | ✅ 一致 |
| **固定数量** | 4个数据中心 | 4个Host | ✅ 一致 |
| **时钟系统** | 全局时钟 | 工作流虚拟时钟 | ⚠️ 不同 |
| **特征维度** | 24维（4DC×6h） | 1维（当前时间） | ⚠️ 简化 |
| **调度模式** | 动态在线 | 离线批处理 | ⚠️ 不同 |
| **特征集成** | 拼接到状态向量 | 作为GNN VM特征 | ⚠️ 不同 |

---

## ⚙️ 配置说明

### 修改碳强度数据

编辑 `scheduler/config/carbon_intensity.py`：

```python
def gen_carbon_intensity_data(num_hosts: int = 4) -> list[list[float]]:
    # 修改这里的数值
    host1 = [0.15, 0.10, 0.11, ...]  # 24个数值
    host2 = [0.07, 0.08, 0.09, ...]
    host3 = [0.10, 0.09, 0.11, ...]
    host4 = [0.13, 0.13, 0.15, ...]
    # ...
```

### 修改Host数量

**注意**：当前固定为4个Host。如需修改，需要：
1. 修改 `FIXED_NUM_HOSTS` 常量
2. 在 `gen_carbon_intensity_data()` 中添加对应的碳强度曲线

---

## 💡 使用示例

### 示例1：获取VM的碳强度

```python
# 在环境中
obs, info = env.reset()
vm_obs = obs.vm_observations[0]

# 获取VM在任务开始时间的碳强度
carbon_intensity = vm_obs.get_carbon_intensity_at(vm_obs.completion_time)
print(f"碳强度: {carbon_intensity}")
```

### 示例2：计算某个任务的碳成本

```python
task_obs = obs.task_observations[task_id]
if task_obs.assigned_vm_id is not None:
    vm_obs = obs.vm_observations[task_obs.assigned_vm_id]
    carbon_intensity = vm_obs.get_carbon_intensity_at(task_obs.start_time)
    carbon_cost = task_obs.energy_consumption * carbon_intensity
    print(f"任务碳成本: {carbon_cost}")
```

### 示例3：计算总碳成本

```python
# 使用预留的接口
total_carbon_cost = obs.carbon_cost()
print(f"总碳成本: {total_carbon_cost}")
```

---

## 🎯 未来扩展方向

### 1. 动态碳强度

目前使用固定的24小时碳强度曲线，可以扩展为：
- 实时碳强度API
- 基于天气的碳强度预测
- 季节性碳强度变化

### 2. 多目标优化

可以在奖励函数中实现：
- Makespan最小化
- 能耗最小化
- 碳成本最小化
- 帕累托最优解

### 3. 碳预算约束

添加硬约束：
```python
if total_carbon_cost > carbon_budget:
    penalty = -1000
```

---

## 📚 参考资料

- 原始项目：`ecmws-experiments/utils/given_data.py`
- 电价处理：`ecmws-experiments/resources/system.py`
- GNN特征：`ecmws-experiments/models/task_embedding.py`

---

## ✨ 总结

本次集成成功实现了：

1. ✅ 碳强度数据管理（4个Host × 24小时）
2. ✅ Host固定数量（4个）
3. ✅ 虚拟时钟系统（工作流级别）
4. ✅ VM特征扩展（添加碳强度）
5. ✅ GNN特征集成（VM节点+1维）
6. ✅ 碳成本计算接口（为奖励函数预留）

**下一步**：修改奖励函数，添加碳成本组件，实现真正的碳感知调度！

