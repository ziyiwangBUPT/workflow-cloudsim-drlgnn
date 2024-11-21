import dataclasses

from typing import Any
from gym_simulator.core.types import VmDto, TaskDto, VmAssignmentDto
from gym_simulator.environments.basic import BasicCloudSimEnvironment


class StaticCloudSimEnvironment(BasicCloudSimEnvironment):
    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)
        del self.observation_space
        del self.action_space

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[tuple[list[TaskDto], list[VmDto]], dict[str, Any]]:
        dict_observation, info = super().reset(seed=seed, options=options)
        tasks = [TaskDto(**task) for task in dict_observation["tasks"]]
        vms = [VmDto(**vm) for vm in dict_observation["vms"]]
        return (tasks, vms), info

    def step(
        self, action: list[VmAssignmentDto]
    ) -> tuple[tuple[list[TaskDto], list[VmDto]], float, bool, bool, dict[str, Any]]:
        dict_action = [dataclasses.asdict(a) for a in action]
        dict_observation, reward, terminated, truncated, info = super().step(dict_action)
        tasks = [TaskDto(**task) for task in dict_observation["tasks"]]
        vms = [VmDto(**vm) for vm in dict_observation["vms"]]
        return (tasks, vms), reward, terminated, truncated, info
