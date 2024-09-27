from dataclasses import dataclass


@dataclass
class VmDto:
    id: int
    memory_mb: int
    cpu_speed_mips: float
    host_power_idle_watt: float
    host_power_peak_watt: float


@dataclass
class TaskDto:
    id: int
    workflow_id: int
    length: int
    req_memory_mb: int
    child_ids: list[int]


@dataclass
class VmAssignmentDto:
    vm_id: int
    workflow_id: int
    task_id: int
