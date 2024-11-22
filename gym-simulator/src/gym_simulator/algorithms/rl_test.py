import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gym_simulator.algorithms.base import BaseScheduler
from gym_simulator.algorithms.rl_agents.gin_agent import GinAgent
from gym_simulator.algorithms.rl_agents.mpgn_agent import MpgnAgent
from gym_simulator.core.simulators.proxy import InternalProxySimulatorObs
from gym_simulator.core.types import TaskDto, VmAssignmentDto, VmDto
from gym_simulator.environments.rl_vm import RlVmCloudSimEnvironment


class RlTestScheduler(BaseScheduler):
    """
    RL Env based scheduler.

    This scheduler runs the RL environment instance internally to schedule the tasks.
    """

    vm_completion_time: np.ndarray | None = None

    def __init__(self, env_config: dict[str, Any], model_type: str, model_path: Path):
        self.model_type = model_type
        self.model_path = model_path
        self.env_config = env_config

    def schedule(self, tasks: list[TaskDto], vms: list[VmDto]) -> list[VmAssignmentDto]:
        if self.model_type == "gin":
            agent = GinAgent(device=torch.device("cpu"))
        else:
            raise NotImplementedError(self.model_type)

        # Load agent and obs state
        agent.load_state_dict(torch.load(str(self.model_path), weights_only=True))
        self.env_config["simulator_kwargs"]["proxy_obs"].tasks = tasks
        self.env_config["simulator_kwargs"]["proxy_obs"].vms = vms

        if self.vm_completion_time is None:
            self.vm_completion_time = np.zeros(len(vms))

        env = RlVmCloudSimEnvironment(env_config=copy.deepcopy(self.env_config))
        env.initial_vm_completion_time = self.vm_completion_time

        next_obs, _ = env.reset(seed=self.env_config["seed"])
        while True:
            # Reshape to have 1 bactch size
            next_obs = next_obs.reshape(1, -1)

            obs_tensor = torch.Tensor(next_obs).to("cpu")
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
            vm_action = action.cpu().numpy()[0]
            next_obs, _, terminated, truncated, info = env.step(vm_action)
            if terminated or truncated:
                break

        self.vm_completion_time = env.state.vm_completion_time.copy()
        assert len(tasks) == len(info["vm_assignments"])
        return info["vm_assignments"]
