from dataclasses import dataclass

from scheduler.dataset_generator.core.models import Vm, Host, Task


@dataclass
class TaskDto:
    id: int
    workflow_id: int
    length: int
    req_memory_mb: int
    child_ids: list[int]

    @staticmethod
    def from_task(task: Task):
        return TaskDto(
            id=task.id,
            workflow_id=task.workflow_id,
            length=task.length,
            req_memory_mb=task.req_memory_mb,
            child_ids=task.child_ids,
        )


@dataclass
class VmDto:
    id: int
    memory_mb: int
    cpu_speed_mips: float
    host_power_idle_watt: float
    host_power_peak_watt: float
    host_cpu_speed_mips: float

    @staticmethod
    def from_vm(vm: Vm, host: Host):
        assert vm.host_id == host.id, "This VM does not belong to the host specified"
        return VmDto(
            id=vm.id,
            memory_mb=vm.memory_mb,
            cpu_speed_mips=vm.cpu_speed_mips,
            host_cpu_speed_mips=host.cpu_speed_mips,
            host_power_idle_watt=host.power_idle_watt,
            host_power_peak_watt=host.power_peak_watt,
        )


@dataclass
class VmAssignmentDto:
    vm_id: int
    workflow_id: int
    task_id: int


TaskIdType = tuple[int, int]
VmIdType = int
