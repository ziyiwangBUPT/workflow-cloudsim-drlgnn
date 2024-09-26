import dataclasses

from typing import Any, override
from gym_simulator.algorithms.types import VmDto, TaskDto, VmAssignmentDto
from gym_simulator.environments.basic import BasicCloudSimEnvironment


class StaticCloudSimEnvironment(BasicCloudSimEnvironment):
    def __init__(self, env_config: dict[str, Any]):
        # Override args
        simulator_kwargs = env_config.get("simulator_kwargs", {})
        simulator_kwargs["dataset_args"] = simulator_kwargs.get("dataset_args", {})
        assert "task_arrival" not in simulator_kwargs["dataset_args"], "task_arrival is set by the environment"
        simulator_kwargs["dataset_args"]["task_arrival"] = "static"

        super().__init__(env_config)
        del self.observation_space
        del self.action_space

    @override
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[tuple[list[TaskDto], list[VmDto]], dict[str, Any]]:
        dict_observation, info = super().reset(seed=seed, options=options)
        tasks = [TaskDto(**task) for task in dict_observation["tasks"]]
        vms = [VmDto(**vm) for vm in dict_observation["vms"]]
        return (tasks, vms), info

    @override
    def step(
        self, action: list[VmAssignmentDto]
    ) -> tuple[tuple[list[TaskDto], list[VmDto]], float, bool, bool, dict[str, Any]]:
        dict_action = [dataclasses.asdict(a) for a in action]
        dict_observation, reward, terminated, truncated, info = super().step(dict_action)
        tasks = [TaskDto(**task) for task in dict_observation["tasks"]]
        vms = [VmDto(**vm) for vm in dict_observation["vms"]]
        return (tasks, vms), reward, terminated, truncated, info
