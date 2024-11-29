import random

from scheduler.dataset_generator.core.gen_task import generate_delay, generate_task_length, generate_dag
from scheduler.dataset_generator.core.models import Task, Workflow


def generate_workflows(
    max_tasks_per_workflow: int,
    num_tasks: int,
    dag_method: str,
    task_length_dist: str,
    min_task_length: int,
    max_task_length: int,
    max_req_memory_mb: int,
    task_arrival: str,
    arrival_rate: float,
) -> list[Workflow]:
    """
    Generate a list of workflows.
    """

    def delay_gen() -> int:
        return int(generate_delay(task_arrival, arrival_rate=arrival_rate))

    def task_length_gen() -> int:
        return int(generate_task_length(task_length_dist, min_task_length, max_task_length))

    def req_memory_gen() -> int:
        return random.randint(1, max_req_memory_mb // 1024) * 1024

    def dag_gen() -> dict[int, set[int]]:
        return generate_dag(dag_method, gnp_min_n=1, gnp_max_n=random.randint(1, max_tasks_per_workflow))

    arrival_time = 0
    workflows: list[Workflow] = []
    generated_task_count: int = 0
    while generated_task_count < num_tasks:
        dag = dag_gen()
        workflow_id = len(workflows)
        tasks: list[Task] = [
            Task(
                id=task_id,
                workflow_id=workflow_id,
                length=task_length_gen(),
                req_memory_mb=req_memory_gen(),
                child_ids=list(child_ids),
            )
            for task_id, child_ids in dag.items()
        ]
        arrival_time += delay_gen()
        workflows.append(Workflow(id=workflow_id, tasks=tasks, arrival_time=arrival_time))
        generated_task_count += len(tasks)

    return workflows
