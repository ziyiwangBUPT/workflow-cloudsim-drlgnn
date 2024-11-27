from scheduler.rl_model.core.types import TaskDto, VmDto
from scheduler.rl_model.core.utils.helpers import active_energy_consumption_per_mi
from scheduler.viz_results.algorithms.base_ready_queue import BaseReadyQueueScheduler


class PowerSavingScheduler(BaseReadyQueueScheduler):
    """
    Implementation of the Power Saving scheduling algorithm.

    Power Saving is a simple scheduling algorithm that schedules the task on the VM with the lowest power consumption per processed MIP.
    This is guaranteed to be the most energy-efficient scheduling algorithm, but it may not be the fastest.
    """

    def select_task(self, ready_tasks: list[TaskDto]) -> TaskDto:
        return ready_tasks[0]

    def select_vm(self, task: TaskDto, vms: list[VmDto]) -> VmDto:
        return min(vms, key=active_energy_consumption_per_mi)
