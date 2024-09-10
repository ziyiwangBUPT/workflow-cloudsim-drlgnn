from dataset_generator.core.models import Workflow, VmAssignment


def print_solution(workflows: list[Workflow], result: list[VmAssignment]):
    makespan_start = {workflow.id: float("inf") for workflow in workflows}
    makespan_end = {workflow.id: float("-inf") for workflow in workflows}

    for assignment in result:
        workflow_id = assignment.workflow_id
        makespan_start[workflow_id] = min(makespan_start[workflow_id], assignment.start)
        makespan_end[workflow_id] = max(makespan_end[workflow_id], assignment.end)
        print(assignment)
    total_makespan = sum(makespan_end[workflow_id] - makespan_start[workflow_id] for workflow_id in makespan_start)

    print()
    print(f"Total makespan: {total_makespan}")
