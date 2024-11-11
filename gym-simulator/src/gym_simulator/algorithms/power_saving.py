from gym_simulator.algorithms.base_ready_queue import BaseReadyQueueScheduler
from gym_simulator.core.types import TaskDto, VmDto


class PowerSavingScheduler(BaseReadyQueueScheduler):
    """
    Implementation of the Power Saving scheduling algorithm.

    Power Saving is a simple scheduling algorithm that schedules the task on the VM with the lowest power consumption per processed MIP.
    This is guaranteed to be the most energy-efficient scheduling algorithm, but it may not be the fastest.
    """

    def select_task(self, ready_tasks: list[TaskDto]) -> TaskDto:
        return ready_tasks[0]

    def select_vm(self, task: TaskDto, vms: list[VmDto]) -> VmDto:
        return min(vms, key=self._power_consumption_per_processed_mip)

    def _power_consumption_per_processed_mip(self, vm: VmDto) -> float:
        host_capacity_frac = vm.cpu_speed_mips / vm.host_cpu_speed_mips
        return vm.host_power_idle_watt + host_capacity_frac * (vm.host_power_peak_watt - vm.host_power_idle_watt)
