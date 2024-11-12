import copy
from pathlib import Path
from typing import Any

import torch
from gym_simulator.algorithms.base import BaseScheduler
from gym_simulator.algorithms.graph.actor import Actor
from gym_simulator.core.types import TaskDto, VmAssignmentDto, VmDto
from gym_simulator.environments.rl_vm import RlVmCloudSimEnvironment


class RlTestScheduler(BaseScheduler):
    """
    RL Env based scheduler.

    This scheduler runs the RL environment instance internally to schedule the tasks.
    """

    def __init__(self, env_config: dict[str, Any], model_dir: str):
        self.model_dir = model_dir
        self.env_config = env_config

    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        env = RlVmCloudSimEnvironment(env_config=copy.deepcopy(self.env_config))
        next_obs, _ = env.reset(seed=self.env_config["seed"])

        agent = Actor(
            max_machines=self.env_config["vm_count"],
            max_jobs=(self.env_config["task_limit"] + 2) * self.env_config["workflow_count"],
        )
        model_path = Path(__file__).parent.parent.parent.parent / "logs" / self.model_dir / "model.pt"
        agent.load_state_dict(torch.load(str(model_path), weights_only=True))
        while True:
            # Reshape to have 1 bactch size
            next_obs = next_obs.reshape(1, -1)

            obs_tensor = torch.Tensor(next_obs).to("cpu")
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            vm_action = action.cpu().numpy()[0]
            next_obs, _, terminated, truncated, info = env.step(vm_action)
            if terminated or truncated:
                break

        assert len(tasks) == len(info["vm_assignments"])
        return info["vm_assignments"]
