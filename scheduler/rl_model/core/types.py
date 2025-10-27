from dataclasses import dataclass

from scheduler.dataset_generator.core.models import Vm, Host, Task


@dataclass
class TaskDto:
    id: int
    workflow_id: int
    length: int
    req_memory_mb: int
    child_ids: list[int]
    deadline: float = 0.0  # 新增：任务的子截止时间

    @staticmethod
    def from_task(task: Task):
        # 安全获取deadline，如果不存在则使用默认值0.0
        deadline = getattr(task, 'deadline', 0.0)
        
        return TaskDto(
            id=task.id,
            workflow_id=task.workflow_id,
            length=task.length,
            req_memory_mb=task.req_memory_mb,
            child_ids=task.child_ids,
            deadline=deadline,  # 新增：传递deadline
        )

    def to_task(self):
        return Task(
            id=self.id,
            workflow_id=self.workflow_id,
            length=self.length,
            req_memory_mb=self.req_memory_mb,
            child_ids=self.child_ids,
            deadline=self.deadline,  # 新增：传递deadline
        )


@dataclass
class VmDto:
    id: int
    memory_mb: int
    cpu_speed_mips: float
    host_power_idle_watt: float
    host_power_peak_watt: float
    host_cpu_speed_mips: float
    host_id: int  # 新增：Host ID，用于获取碳强度
    host_carbon_intensity_curve: list[float]  # 新增：Host的24小时碳强度曲线

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
            host_id=host.id,  # 新增
            host_carbon_intensity_curve=host.carbon_intensity_curve or [0.1] * 24,  # 新增
        )
    
    def get_carbon_intensity_at(self, time_seconds: float) -> float:
        """获取指定时间的碳强度值"""
        hour = int(time_seconds // 3600) % 24
        return self.host_carbon_intensity_curve[hour]

    def to_vm(self):
        return Vm(
            id=self.id,
            host_id=self.id,
            cpu_speed_mips=int(self.cpu_speed_mips),
            memory_mb=self.memory_mb,
        )

    def to_host(self):
        return Host(
            id=self.id,
            cores=1,
            cpu_speed_mips=int(self.host_cpu_speed_mips),
            power_idle_watt=int(self.host_power_idle_watt),
            power_peak_watt=int(self.host_power_peak_watt),
        )


@dataclass
class VmAssignmentDto:
    vm_id: int
    workflow_id: int
    task_id: int


TaskIdType = tuple[int, int]
VmIdType = int
