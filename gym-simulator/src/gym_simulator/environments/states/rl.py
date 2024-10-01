from dataclasses import dataclass

import numpy as np

from gym_simulator.core.types import TaskDto
from gym_simulator.utils.task_mapper import TaskMapper


@dataclass
class RlEnvState:
    task_mapper: TaskMapper
    task_state_scheduled: np.ndarray
    task_state_ready: np.ndarray
    task_completion_time: np.ndarray
    vm_completion_time: np.ndarray
    task_vm_compatibility: np.ndarray
    task_vm_time_cost: np.ndarray
    task_vm_power_cost: np.ndarray
    task_graph_edges: np.ndarray
    assignments: np.ndarray
    tasks: list[TaskDto]

    def to_observation(self) -> dict[str, np.ndarray]:
        return {
            "task_state_scheduled": self.task_state_scheduled,
            "task_state_ready": self.task_state_ready,
            "task_completion_time": self.task_completion_time,
            "vm_completion_time": self.vm_completion_time,
            "task_vm_compatibility": self.task_vm_compatibility,
            "task_vm_time_cost": self.task_vm_time_cost,
            "task_vm_power_cost": self.task_vm_power_cost,
            "task_graph_edges": self.task_graph_edges,
        }
