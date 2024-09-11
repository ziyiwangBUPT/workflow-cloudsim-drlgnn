from dataclasses import dataclass


@dataclass
class Task:
    id: int
    workflow_id: int
    length: int
    req_cores: int
    child_ids: list[int]


@dataclass
class Workflow:
    id: int
    tasks: list[Task]
    arrival_time: int

    @staticmethod
    def from_json(data: dict) -> "Workflow":
        tasks = [Task(**task) for task in data.pop("tasks")]
        return Workflow(tasks=tasks, **data)


@dataclass
class Vm:
    id: int
    host_id: int
    cores: int
    cpu_speed_mips: int
    memory_mb: int
    disk_mb: int
    bandwidth_mbps: int
    vmm: str


@dataclass
class Host:
    id: int
    cores: int
    cpu_speed_mips: int
    memory_mb: int
    disk_mb: int
    bandwidth_mbps: int
    power_idle_watt: int
    power_peak_watt: int


@dataclass
class VmAssignment:
    workflow_id: int
    task_id: int
    vm_id: int
    start_time: int
    end_time: int


@dataclass
class Dataset:
    workflows: list[Workflow]
    vms: list[Vm]
    hosts: list[Host]

    @staticmethod
    def from_json(data: dict) -> "Dataset":
        workflows = [Workflow.from_json(workflow) for workflow in data.pop("workflows")]
        vms = [Vm(**vm) for vm in data.pop("vms")]
        hosts = [Host(**host) for host in data.pop("hosts")]
        return Dataset(workflows=workflows, vms=vms, hosts=hosts)


@dataclass
class Solution:
    dataset: Dataset
    vm_assignments: list[VmAssignment]

    @staticmethod
    def from_json(data: dict) -> "Solution":
        dataset = Dataset.from_json(data.pop("dataset"))
        vm_assignments = [VmAssignment(**vm_assignment) for vm_assignment in data.pop("vm_assignments")]
        return Solution(dataset=dataset, vm_assignments=vm_assignments)
