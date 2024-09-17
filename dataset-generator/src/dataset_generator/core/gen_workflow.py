import random

from dataset_generator.core.models import Task, Workflow
from dataset_generator.core.gen_task import generate_task_length, generate_dag, generate_poisson_delay


def generate_workflows(
    workflow_count: int,
    dag_method: str,
    gnp_min_n: int,
    gnp_max_n: int,
    task_length_dist: str,
    min_task_length: int,
    max_task_length: int,
    max_req_cores: int,
    arrival_rate: float,
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

    def dag_gen() -> dict[int, set[int]]:
        return generate_dag(dag_method, gnp_min_n=gnp_min_n, gnp_max_n=gnp_max_n)

    arrival_time = 0
    workflows: list[Workflow] = []
    for workflow_id in range(workflow_count):
        dag = dag_gen()
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
