import random

from dataset_generator.core.models import Task, Workflow
from dataset_generator.core.gen_task import generate_task_length, generate_dag, generate_poisson_delay


def generate_workflows(
    workflow_count: int,
    min_task_count: int,
    max_task_count: int,
    task_length_dist: str,
    min_task_length: int,
    max_task_length: int,
    max_req_cores: int,
    arrival_rate: int,
) -> list[Workflow]:
    """
    Generate a list of workflows.
    """

    def delay_gen() -> int:
        return int(generate_poisson_delay(arrival_rate))

    def task_length_gen() -> int:
        return int(generate_task_length(task_length_dist, min_task_length, max_task_length))

    def req_cores_gen() -> int:
        return random.randint(1, max_req_cores)

    def dag_gen(n: int) -> dict[int, set[int]]:
        return generate_dag(n)

    arrival_time = 0
    workflows: list[Workflow] = []
    for workflow_id in range(workflow_count):
        task_count = random.randint(min_task_count, max_task_count)
        dag = dag_gen(task_count)
        tasks: list[Task] = [
            Task(
                id=task_id,
                workflow_id=workflow_id,
                length=task_length_gen(),
                req_cores=req_cores_gen(),
                child_ids=list(child_ids),
            )
            for task_id, child_ids in dag.items()
        ]
        arrival_time += delay_gen()
        workflows.append(Workflow(id=workflow_id, tasks=tasks, arrival_time=arrival_time))

    return workflows
