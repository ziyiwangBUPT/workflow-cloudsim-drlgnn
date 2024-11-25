import torch
import numpy as np

from scheduler.dataset_generator.core.models import Dataset, Workflow, Task, Vm, Host
from scheduler.rl_model.agents.gin_e_agent.agent import GinEAgent
from scheduler.rl_model.agents.gin_e_agent.wrapper import GinEAgentWrapper
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.core.types import TaskDto, VmDto, VmAssignmentDto
from scheduler.viz_results.algorithms.base import BaseScheduler


class GinEAgentScheduler(BaseScheduler):
    vm_completion_time: np.ndarray | None = None

    def __init__(self, model_path: str):
        self.model_path = model_path

    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        agent = GinEAgent(device=torch.device("cpu"))
        agent.load_state_dict(torch.load(str(self.model_path), weights_only=True))

        if self.vm_completion_time is None:
            self.vm_completion_time = np.zeros(len(vms))

        # Create a fake dataset to be used by the RL environment
        workflow_ds: dict[int, Workflow] = {}
        for task in tasks:
            if task.workflow_id not in workflow_ds:
                workflow_ds[task.workflow_id] = Workflow(id=task.workflow_id, tasks=[], arrival_time=0)
            workflow_ds[task.workflow_id].tasks.append(
                Task(
                    id=task.id,
                    workflow_id=task.workflow_id,
                    length=task.length,
                    req_memory_mb=task.req_memory_mb,
                    child_ids=task.child_ids,
                )
            )
        vm_ds = [
            Vm(
                id=vm.id,
                host_id=vm.id,
                cpu_speed_mips=int(vm.cpu_speed_mips),
                memory_mb=vm.memory_mb,
            )
            for vm in vms
        ]
        host_ds = [
            Host(
                id=vm.id,
                cores=1,
                cpu_speed_mips=int(vm.host_cpu_speed_mips),
                power_idle_watt=int(vm.host_power_idle_watt),
                power_peak_watt=int(vm.host_power_peak_watt),
            )
            for vm in vms
        ]

        env = GinEAgentWrapper(
            CloudSchedulingGymEnvironment(
                dataset=Dataset(
                    workflows=list(workflow_ds.values()),
                    vms=vm_ds,
                    hosts=host_ds,
                )
            )
        )

        # Run environment
        obs, info = env.reset(seed=0)
        while True:
            tensor_obs = torch.Tensor(obs).reshape(1, -1)
            action, *_ = agent.get_action_and_value(tensor_obs)
            obs, reward, terminated, truncated, info = env.step(int(action.item()))
            if terminated or truncated:
                break

        assert len(tasks) == len(info["assignments"])
        return info["assignments"]
