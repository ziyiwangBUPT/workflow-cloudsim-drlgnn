from dataclasses import dataclass


@dataclass
class Task:
    id: int
    length: int
    req_cores: int
    child_ids: list[int]


@dataclass
class Workflow:
    id: int
    tasks: list[Task]
    arrival_time: int


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
    start: int
    end: int


@dataclass
class Dataset:
    workflows: list[Workflow]
    vms: list[Vm]
    hosts: list[Host]
