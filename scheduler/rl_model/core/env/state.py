from dataclasses import dataclass

from scheduler.rl_model.core.types import TaskDto, VmDto
from scheduler.rl_model.core.utils.task_mapper import TaskMapper
from scheduler.rl_model.core.env.clock_manager import ClockManager


@dataclass
class EnvState:
    static_state: "StaticState"
    task_states: list["TaskState"]
    vm_states: list["VmState"]
    task_dependencies: set[tuple[int, int]]
    clock_manager: ClockManager | None = None  # 新增：虚拟时钟管理器


@dataclass
class VmState:
    assigned_task_id: int | None = None
    completion_time: float = 0


@dataclass
class TaskState:
    is_ready: bool = False
    assigned_vm_id: int | None = None
    start_time: float = 0
    completion_time: float = 0
    energy_consumption: float = 0
    carbon_cost: float = 0  # 新增：任务的碳成本（gCO2）


@dataclass
class StaticState:
    task_mapper: TaskMapper
    tasks: list[TaskDto]
    vms: list[VmDto]
    compatibilities: list[tuple[int, int]]
