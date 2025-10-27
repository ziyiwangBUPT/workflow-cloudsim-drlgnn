"""
调试测试环境创建
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scheduler.dataset_generator.core.gen_dataset import generate_dataset
from scheduler.rl_model.core.types import TaskDto


def debug_task_mapping():
    """调试任务映射过程"""
    print("\n" + "=" * 60)
    print("调试任务映射")
    print("=" * 60)
    
    # 生成数据集
    dataset = generate_dataset(
        seed=42,
        host_count=4,
        vm_count=12,
        workflow_count=2,
        gnp_min_n=5,
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
    
    print(f"\n✓ 生成了 {len(dataset.workflows)} 个工作流")
    
    # 显示每个工作流的任务
    for workflow in dataset.workflows:
        print(f"\n工作流 {workflow.id}:")
        print(f"  任务数量: {len(workflow.tasks)}")
        for task in workflow.tasks[:5]:  # 只显示前5个
            print(f"    任务 {task.id}: 长度={task.length}, 子任务={task.child_ids[:3]}...")
    
    # 转换为TaskDto
    tasks = [TaskDto.from_task(task) for workflow in dataset.workflows for task in workflow.tasks]
    print(f"\n✓ 转换为TaskDto: {len(tasks)} 个任务")
    
    # 显示转换后的任务ID
    print("\n转换后的任务ID和workflow_id:")
    for task in tasks[:10]:  # 只显示前10个
        print(f"  任务 {task.id}: workflow_id={task.workflow_id}, length={task.length}")
    
    return dataset


def debug_taskmapper():
    """调试TaskMapper"""
    from scheduler.rl_model.core.utils.task_mapper import TaskMapper
    
    dataset = debug_task_mapping()
    
    # 从数据集创建任务
    tasks = [TaskDto.from_task(task) for workflow in dataset.workflows for task in workflow.tasks]
    
    print("\n" + "=" * 60)
    print("TaskMapper 映射")
    print("=" * 60)
    
    # 创建TaskMapper
    task_mapper = TaskMapper(tasks)
    print(f"\n✓ TaskMapper创建成功")
    print(f"  task_counts_cum: {task_mapper._task_counts_cum}")
    
    # 映射任务
    mapped_tasks = task_mapper.map_tasks()
    print(f"\n✓ 映射完成: {len(mapped_tasks)} 个任务（包括dummy）")
    
    # 显示映射结果
    print("\n映射后的任务:")
    for i, task in enumerate(mapped_tasks[:15]):  # 只显示前15个
        print(f"  索引 {i}: ID={task.id}, workflow_id={task.workflow_id}, length={task.length}")
    
    # 检查ID连续性
    print("\n检查ID连续性:")
    for i, task in enumerate(mapped_tasks):
        if task.id != i:
            print(f"  ❌ 不匹配: 索引={i}, ID={task.id}")
            # 找出应该是什么
            if i < len(mapped_tasks):
                print(f"     下一个任务: ID={mapped_tasks[i+1].id if i+1<len(mapped_tasks) else 'N/A'}")
            break
        elif i < 10:
            print(f"  ✓ 索引 {i}: ID={task.id}")


if __name__ == "__main__":
    try:
        debug_taskmapper()
    except Exception as e:
        print(f"\n❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

