from dataclasses import dataclass
import torch


@dataclass
class DecodedObservation:
    task_state_scheduled: torch.Tensor
    task_state_ready: torch.Tensor
    task_completion_time: torch.Tensor
    vm_completion_time: torch.Tensor
    task_vm_compatibility: torch.Tensor
    task_vm_time_cost: torch.Tensor
    task_vm_power_cost: torch.Tensor
    task_graph_edges: torch.Tensor


def decode_observation(x: torch.Tensor):
    n_jobs = int(x[0].long().item())
    n_machines = int(x[1].long().item())
    x = x[2:]

    task_state_scheduled = x[:n_jobs].long()
    x = x[n_jobs:]

    task_state_ready = x[:n_jobs].long()
    x = x[n_jobs:]

    task_completion_time = x[:n_jobs]
    x = x[n_jobs:]

    vm_completion_time = x[:n_machines]
    x = x[n_machines:]

    task_vm_compatibility = x[: n_jobs * n_machines].reshape(n_jobs, n_machines).long()
    x = x[n_jobs * n_machines :]

    task_vm_time_cost = x[: n_jobs * n_machines].reshape(n_jobs, n_machines)
    x = x[n_jobs * n_machines :]

    task_vm_power_cost = x[: n_jobs * n_machines].reshape(n_jobs, n_machines)
    x = x[n_jobs * n_machines :]

    task_graph_edges = x[: n_jobs * n_jobs].reshape(n_jobs, n_jobs).long()
    x = x[n_jobs * n_jobs :]

    return DecodedObservation(
        task_state_scheduled=task_state_scheduled,
        task_state_ready=task_state_ready,
        task_completion_time=task_completion_time,
        vm_completion_time=vm_completion_time,
        task_vm_compatibility=task_vm_compatibility,
        task_vm_time_cost=task_vm_time_cost,
        task_vm_power_cost=task_vm_power_cost,
        task_graph_edges=task_graph_edges,
    )
