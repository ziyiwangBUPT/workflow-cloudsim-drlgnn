from dataclasses import dataclass

from scheduler.rl_model.core.types import TaskDto, VmDto, VmAssignmentDto


@dataclass
class SimEnvObservation:
    tasks: list[TaskDto]
    vms: list[VmDto]


@dataclass
class SimEnvAction:
    vm_assignments: list[VmAssignmentDto]
