import dataclasses
from dataclasses import dataclass


@dataclass
class Task:
    id: int
    workflow_id: int
    length: int
    req_memory_mb: int
    child_ids: list[int]
    
    # 预调度阶段新增属性
    avg_est: float = 0.0      # 任务在平均资源上的估计最早开始时间
    avg_eft: float = 0.0      # 任务在平均资源上的估计最早完成时间
    rank_dp: float = 0.0      # 任务的DP排名（用作任务内优先级分数）
    deadline: float = 0.0     # 任务的子截止时间
    global_priority: float = 0.0  # 全局优先级分数（工作流优先级 × 任务优先级）
    
    # 用于算法计算的辅助属性
    parent_ids: list[int] = None  # 父任务ID列表（从child_ids反推）
    max_avg_transtime: float = 0.0  # 平均最大传输时间
    
    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []


@dataclass
class Workflow:
    id: int
    tasks: list[Task]
    arrival_time: int
    
    # 预调度阶段新增属性
    avg_eft: float = 0.0        # 工作流在平均资源上的估计最早完成时间
    avg_slacktime: float = 0.0  # 工作流的平均松弛时间 (deadline - avg_eft)
    workload: float = 0.0       # 工作流的总计算负载 (所有任务负载之和)
    deadline: float = 0.0       # 工作流的截止时间
    workflow_priority: float = 0.0  # 工作流优先级分数（WS算法计算，值越小优先级越高）
    
    # 虚拟时钟管理（新增）
    virtual_clock: float = 0.0  # 工作流的虚拟时钟（秒），从0开始
    
    def advance_clock(self, time_delta: float) -> None:
        """推进工作流的虚拟时钟"""
        self.virtual_clock += time_delta
    
    def get_current_hour(self) -> int:
        """获取当前虚拟时钟对应的小时（0-23）"""
        return int(self.virtual_clock // 3600) % 24

    @staticmethod
    def from_json(data: dict) -> "Workflow":
        tasks = [Task(**task) for task in data.pop("tasks")]
        return Workflow(tasks=tasks, **data)


@dataclass
class Vm:
    id: int
    host_id: int
    cpu_speed_mips: int
    memory_mb: int
    disk_mb: int = -1
    bandwidth_mbps: int = -1
    vmm: str = "Xen"


@dataclass
class Host:
    id: int
    cores: int
    cpu_speed_mips: int
    power_idle_watt: int
    power_peak_watt: int
    memory_mb: int = -1
    disk_mb: int = -1
    bandwidth_mbps: int = -1
    carbon_intensity_curve: list[float] = None  # 新增：24小时碳强度曲线
    
    def get_carbon_intensity_at(self, time_seconds: float) -> float:
        """获取指定时间的碳强度值"""
        if self.carbon_intensity_curve is None:
            return 0.1  # 默认值
        hour = int(time_seconds // 3600) % 24
        return self.carbon_intensity_curve[hour]


@dataclass
class VmAssignment:
    workflow_id: int
    task_id: int
    vm_id: int
    start_time: float
    end_time: float


@dataclass
class Dataset:
    workflows: list[Workflow]
    vms: list[Vm]
    hosts: list[Host]

    def to_json(self):
        # noinspection PyTypeChecker
        return dataclasses.asdict(self)

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

    def to_json(self):
        # noinspection PyTypeChecker
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(data: dict) -> "Solution":
        dataset = Dataset.from_json(data.pop("dataset"))
        vm_assignments = [VmAssignment(**vm_assignment) for vm_assignment in data.pop("vm_assignments")]
        return Solution(dataset=dataset, vm_assignments=vm_assignments)
