"""
验证Task ID修复

使用与gym_env相同的逻辑，验证修复是否有效
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scheduler.dataset_generator.core.gen_dataset import generate_dataset
from scheduler.rl_model.core.types import TaskDto
from scheduler.rl_model.core.utils.task_mapper import TaskMapper
from scheduler.pre_scheduling.pre_computation import precompute_workflow_data
from scheduler.pre_scheduling.ws_method import ContentionAwareWorkflowSequencing
from scheduler.pre_scheduling.dp_method import BottleLayerAwareDeadlinePartition


# 生成数据集
dataset = generate_dataset(
    seed=42,
    host_count=4,
    vm_count=10,
    workflow_count=5,
    gnp_min_n=10,
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

print(f"生成了 {len(dataset.workflows)} 个工作流")
print(f"原始工作流ID: {[wf.id for wf in dataset.workflows]}")

# 预调度
ws_scheduler = ContentionAwareWorkflowSequencing(alpha1=0.33, alpha2=0.33, alpha3=0.33)
dp_scheduler = BottleLayerAwareDeadlinePartition(beta=0.5)
rho = 0.2

for workflow in dataset.workflows:
    precompute_workflow_data(workflow, dataset.vms, rho)

sorted_workflows = ws_scheduler.run(dataset.workflows, dataset.vms)
print(f"WS排序后工作流ID: {[wf.id for wf in sorted_workflows]}")

for workflow in sorted_workflows:
    dp_scheduler.run(workflow, dataset.vms)

# 🔧 应用修复：重新分配workflow_id
print("\n应用修复：重新分配workflow_id")
for new_wf_id, workflow in enumerate(sorted_workflows):
    old_wf_id = workflow.id
    workflow.id = new_wf_id
    for task in workflow.tasks:
        task.workflow_id = new_wf_id
    print(f"  工作流: {old_wf_id} → {new_wf_id}")

print(f"\n修复后工作流ID: {[wf.id for wf in sorted_workflows]}")

dataset.workflows = sorted_workflows

# 转换和映射
tasks = [TaskDto.from_task(task) for workflow in dataset.workflows for task in workflow.tasks]
task_mapper = TaskMapper(tasks)
mapped_tasks = task_mapper.map_tasks()

print(f"\n映射后任务数: {len(mapped_tasks)}")
print(f"_task_counts_cum: {task_mapper._task_counts_cum}")

# 检查ID连续性
print("\n检查ID连续性:")
errors = []
for i, task in enumerate(mapped_tasks):
    if task.id != i:
        errors.append((i, task.id))
        if len(errors) <= 5:
            print(f"  ❌ 索引 {i}: ID={task.id} (期望{i})")
    elif i < 10:
        print(f"  ✓ 索引 {i}: ID={task.id}")

if not errors:
    print("\n✅ 所有任务ID连续！修复成功！")
else:
    print(f"\n❌ 发现 {len(errors)} 个ID不匹配")

