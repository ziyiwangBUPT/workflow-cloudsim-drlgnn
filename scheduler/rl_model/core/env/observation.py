import copy
from dataclasses import dataclass

from scheduler.rl_model.core.env.state import EnvState


@dataclass
class EnvObservation:
    task_observations: list["TaskObservation"]
    vm_observations: list["VmObservation"]
    task_dependencies: list[tuple[int, int]]
    compatibilities: list[tuple[int, int]]

    def __init__(self, state: EnvState):
        self.task_observations = [
            TaskObservation(
                is_ready=state.task_states[task_id].is_ready,
                assigned_vm_id=state.task_states[task_id].assigned_vm_id,
                start_time=state.task_states[task_id].start_time,
                completion_time=state.task_states[task_id].completion_time,
                energy_consumption=state.task_states[task_id].energy_consumption,
                length=state.static_state.tasks[task_id].length,
            )
            for task_id in range(len(state.task_states))
        ]
        self.vm_observations = [
            VmObservation(
                assigned_task_id=state.vm_states[vm_id].assigned_task_id,
                completion_time=state.vm_states[vm_id].completion_time,
                cpu_speed_mips=state.static_state.vms[vm_id].cpu_speed_mips,
                host_power_idle_watt=state.static_state.vms[vm_id].host_power_idle_watt,
                host_power_peak_watt=state.static_state.vms[vm_id].host_power_peak_watt,
                host_cpu_speed_mips=state.static_state.vms[vm_id].host_cpu_speed_mips,
            )
            for vm_id in range(len(state.vm_states))
        ]
        self.task_dependencies = copy.deepcopy(list(state.task_dependencies))
        self.compatibilities = copy.deepcopy(list(state.static_state.compatibilities))


@dataclass
class TaskObservation:
    is_ready: bool
    assigned_vm_id: int | None
    start_time: float
    completion_time: float
    energy_consumption: float
    length: float


@dataclass
class VmObservation:
    assigned_task_id: int | None
    completion_time: float
    cpu_speed_mips: float
    host_power_idle_watt: float
    host_power_peak_watt: float
    host_cpu_speed_mips: float
