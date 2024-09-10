from dataset_generator.core.models import Dataset
from dataset_generator.core.gen_workflow import generate_workflows
from dataset_generator.core.gen_vm import generate_hosts, generate_vms, allocate_vms


def generate_dataset(
    host_count: int,
    vm_count: int,
    max_cores: int,
    min_cpu_speed_mips: int,
    max_cpu_speed_mips: int,
    workflow_count: int,
    min_task_count: int,
    max_task_count: int,
    task_length_dist: str,
    min_task_length: int,
    max_task_length: int,
    arrival_rate: int,
) -> Dataset:
    """
    Generate a dataset.
    """

    hosts = generate_hosts(host_count)
    vms = generate_vms(vm_count, max_cores, min_cpu_speed_mips, max_cpu_speed_mips)
    vm_allocations = allocate_vms(vms, hosts)

    workflows = generate_workflows(
        workflow_count=workflow_count,
        min_task_count=min_task_count,
        max_task_count=max_task_count,
        task_length_dist=task_length_dist,
        min_task_length=min_task_length,
        max_task_length=max_task_length,
        # Make sure that the problem is feasible
        max_req_cores=max(vm.cores for vm in vms),
        arrival_rate=arrival_rate,
    )

    return Dataset(workflows=workflows, vms=vms, hosts=hosts, vm_allocations=vm_allocations)
