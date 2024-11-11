from dataclasses import dataclass
import random
from typing import Any, Callable
from dataset_generator.core.gen_dataset import generate_dataset
from dataset_generator.core.models import Dataset
from gym_simulator.core.simulators.base import BaseSimulator
from dataset_generator.gen_dataset import Args as DatasetArgs


class InternalSimulator(BaseSimulator):
    dataset_seed: int | None
    current_dataset: Dataset | None
    running = False

    def __init__(self, dataset_args: dict[str, Any]) -> None:
        self.dataset_args = dataset_args

    def start(self):
        assert not self.is_running()
        default_args = DatasetArgs()
        self.current_dataset = generate_dataset(
            seed=random.randint(1, 2**31) if self.dataset_seed is None else self.dataset_seed,
            host_count=self.dataset_args.get("host_count", default_args.host_count),
            vm_count=self.dataset_args.get("vm_count", default_args.vm_count),
            max_memory_gb=self.dataset_args.get("max_memory_gb", default_args.max_memory_gb),
            min_cpu_speed_mips=self.dataset_args.get("min_cpu_speed", default_args.min_cpu_speed),
            max_cpu_speed_mips=self.dataset_args.get("max_cpu_speed", default_args.max_cpu_speed),
            workflow_count=self.dataset_args.get("workflow_count", default_args.workflow_count),
            dag_method=self.dataset_args.get("dag_method", default_args.dag_method),
            gnp_min_n=self.dataset_args.get("gnp_min_n", default_args.gnp_min_n),
            gnp_max_n=self.dataset_args.get("gnp_max_n", default_args.gnp_max_n),
            task_length_dist=self.dataset_args.get("task_length_dist", default_args.task_length_dist),
            min_task_length=self.dataset_args.get("min_task_length", default_args.min_task_length),
            max_task_length=self.dataset_args.get("max_task_length", default_args.max_task_length),
            task_arrival=self.dataset_args.get("task_arrival", default_args.task_arrival),
            arrival_rate=self.dataset_args.get("arrival_rate", default_args.arrival_rate),
        )
        self.running = True

    def stop(self) -> str | None:
        assert self.is_running()
        self.running = False
        return None

    def is_running(self) -> bool:
        return self.running

    def reset(self, seed: int | None) -> Any:
        if self.is_running():
            self.stop()
        self.dataset_seed = seed
        self.start()
        assert self.current_dataset is not None
        return _SimInput(dataset=self.current_dataset).getObservation()

    def step(self, action_creator: Callable[[Any], Any]) -> "_SimInput":
        assert self.is_running()
        action = action_creator(None)
        assert self.current_dataset is not None
        return _SimInput(dataset=self.current_dataset)

    def reboot(self):
        if self.is_running():
            self.stop()
        self.start()


@dataclass
class _SimInput:
    dataset: Dataset

    def getObservation(self):
        hosts = {host.id: host for host in self.dataset.hosts}
        return {
            "tasks": [
                {
                    "id": task.id,
                    "workflow_id": task.workflow_id,
                    "length": task.length,
                    "req_memory_mb": task.req_memory_mb,
                    "child_ids": task.child_ids,
                }
                for workflow in self.dataset.workflows
                for task in workflow.tasks
            ],
            "vms": [
                {
                    "id": vm.id,
                    "memory_mb": vm.memory_mb,
                    "cpu_speed_mips": vm.cpu_speed_mips,
                    "host_power_idle_watt": hosts[vm.host_id].power_idle_watt,
                    "host_power_peak_watt": hosts[vm.host_id].power_peak_watt,
                    "host_cpu_speed_mips": hosts[vm.host_id].cpu_speed_mips,
                }
                for vm in self.dataset.vms
            ],
        }

    def getReward(self):
        return 0

    def isTerminated(self):
        return True

    def isTruncated(self):
        return False

    def getInfo(self):
        return {}
