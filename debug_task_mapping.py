"""
调试Task ID映射问题
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scheduler.dataset_generator.core.gen_dataset import generate_dataset
from scheduler.rl_model.core.types import TaskDto
from scheduler.rl_model.core.utils.task_mapper import TaskMapper
from scheduler.rl_model.core.utils.helpers import is_suitable


def debug_task_mapping():
    """调试任务映射"""
    print("\n" + "=" * 60)
    print("调试任务映射")
    print("=" * 60)
    
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
    
    print(f"✓ 生成了 {len(dataset.workflows)} 个工作流")
    
    # 检查原始任务
    all_tasks = []
    for workflow in dataset.workflows:
        print(f"\n工作流 {workflow.id}: {len(workflow.tasks)} 个任务")
        for task in workflow.tasks[:3]:  # 只显示前3个
            print(f"  任务 {task.id}: workflow_id={workflow.id}, has_deadline={hasattr(task, 'deadline')}")
            all_tasks.append(task)
    
    # 转换为TaskDto
    tasks = [TaskDto.from_task(task) for workflow in dataset.workflows for task in workflow.tasks]
    print(f"\n✓ 转换为TaskDto: {len(tasks)} 个任务")
    
    # 创建TaskMapper
    task_mapper = TaskMapper(tasks)
    
    # 映射任务
    mapped_tasks = task_mapper.map_tasks()
    print(f"\n✓ 映射完成: {len(mapped_tasks)} 个任务（包括dummy）")
    
    # 检查映射结果
    print("\n映射后的前20个任务:")
    for i, task in enumerate(mapped_tasks[:20]):
        print(f"  索引 {i}: ID={task.id}, workflow_id={task.workflow_id}")
    
    # 检查ID连续性
    print("\n检查ID连续性:")
    for i, task in enumerate(mapped_tasks):
        if task.id != i:
            print(f"  ❌ 不匹配: 索引={i}, ID={task.id}")
            if i < 5:
                # 显示前后几个
                start = max(0, i-2)
                end = min(len(mapped_tasks), i+3)
                for j in range(start, end):
                    print(f"     索引 {j}: ID={mapped_tasks[j].id}")
            break
        elif i < 10:
            print(f"  ✓ 索引 {i}: ID={task.id}")


if __name__ == "__main__":
    try:
        debug_task_mapping()
    except Exception as e:
        print(f"\n❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

