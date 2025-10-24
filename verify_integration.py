"""
简单验证脚本 - 检查预调度模块是否正确安装
"""

print("开始验证预调度模块...")

# 测试 1: 导入检查
print("\n1. 检查模块导入...")
try:
    from scheduler.dataset_generator.core.models import Task, Workflow
    print("   ✓ models 导入成功")
except Exception as e:
    print(f"   ✗ models 导入失败: {e}")
    exit(1)

try:
    from scheduler.pre_scheduling.methods_proto import WorkflowSequencing, DeadlinePartition
    print("   ✓ methods_proto 导入成功")
except Exception as e:
    print(f"   ✗ methods_proto 导入失败: {e}")
    exit(1)

try:
    from scheduler.pre_scheduling.pre_computation import precompute_workflow_data
    print("   ✓ pre_computation 导入成功")
except Exception as e:
    print(f"   ✗ pre_computation 导入失败: {e}")
    exit(1)

try:
    from scheduler.pre_scheduling.ws_method import ContentionAwareWorkflowSequencing
    print("   ✓ ws_method 导入成功")
except Exception as e:
    print(f"   ✗ ws_method 导入失败: {e}")
    exit(1)

try:
    from scheduler.pre_scheduling.dp_method import BottleLayerAwareDeadlinePartition
    print("   ✓ dp_method 导入成功")
except Exception as e:
    print(f"   ✗ dp_method 导入失败: {e}")
    exit(1)

# 测试 2: 检查数据模型属性
print("\n2. 检查数据模型新增属性...")
task = Task(id=0, workflow_id=0, length=100, req_memory_mb=1024, child_ids=[])
print(f"   ✓ Task 对象创建成功")
print(f"   - avg_est: {task.avg_est}")
print(f"   - avg_eft: {task.avg_eft}")
print(f"   - rank_dp: {task.rank_dp}")
print(f"   - deadline: {task.deadline}")
print(f"   - parent_ids: {task.parent_ids}")

workflow = Workflow(id=0, tasks=[task], arrival_time=0)
print(f"   ✓ Workflow 对象创建成功")
print(f"   - avg_eft: {workflow.avg_eft}")
print(f"   - avg_slacktime: {workflow.avg_slacktime}")
print(f"   - workload: {workflow.workload}")
print(f"   - deadline: {workflow.deadline}")

print("\n✓ 所有验证通过！预调度模块已成功集成。")

