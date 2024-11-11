import socket
import json

from gymnasium import spaces
from typing import Any, override
import numpy as np

from dataset_generator.core.models import Solution
from gym_simulator.core.environments.cloudsim import BaseCloudSimEnvironment
from gym_simulator.core.simulators.embedded import EmbeddedSimulator
from gym_simulator.core.simulators.internal import InternalSimulator
from gym_simulator.core.simulators.remote import RemoteSimulator


class BasicCloudSimEnvironment(BaseCloudSimEnvironment):
    metadata = {"render_modes": []}

    def __init__(self, env_config: dict[str, Any]):
        self.host_count: int = env_config["host_count"]
        self.vm_count: int = env_config["vm_count"]
        self.workflow_count: int = env_config["workflow_count"]
        self.task_limit: int = env_config["task_limit"]

        self.action_space = spaces.Sequence(
            spaces.Dict(
                {
                    "vm_id": spaces.Discrete(self.vm_count),
                    "workflow_id": spaces.Discrete(self.workflow_count),
                    "task_id": spaces.Discrete(self.task_limit),
                }
            )
        )
        self.observation_space = spaces.Dict(
            {
                "tasks": spaces.Sequence(
                    spaces.Dict(
                        {
                            "id": spaces.Discrete(self.task_limit),
                            "workflow_id": spaces.Discrete(self.workflow_count),
                            "length": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int64),
                            "req_memory_mb": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int64),
                            "child_ids": spaces.Sequence(spaces.Discrete(self.task_limit)),
                        }
                    )
                ),
                "vms": spaces.Sequence(
                    spaces.Dict(
                        {
                            "id": spaces.Discrete(self.vm_count),
                            "memory_mb": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int64),
                            "cpu_speed_mips": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
                            "host_power_idle_watt": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
                            "host_power_peak_watt": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
                            "host_cpu_speed_mips": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
                        }
                    )
                ),
            }
        )

        # Initialize the simulator
        simulator_mode = env_config["simulator_mode"]
        simulator_kwargs = env_config.get("simulator_kwargs", {})
        if simulator_mode == "embedded" or simulator_mode == "internal":
            # Set dataset args
            simulator_kwargs["dataset_args"] = simulator_kwargs.get("dataset_args", {})
            assert "host_count" not in simulator_kwargs["dataset_args"], "host_count is set by the environment"
            assert "vm_count" not in simulator_kwargs["dataset_args"], "vm_count is set by the environment"
            assert "workflow_count" not in simulator_kwargs["dataset_args"], "workflow_count is set by the environment"
            assert "dag_method" not in simulator_kwargs["dataset_args"], "dag_method is set by the environment"
            assert "gnp_max_n" not in simulator_kwargs["dataset_args"], "gnp_max_n is set by the environment"
            simulator_kwargs["dataset_args"]["host_count"] = self.host_count
            simulator_kwargs["dataset_args"]["vm_count"] = self.vm_count
            simulator_kwargs["dataset_args"]["workflow_count"] = self.workflow_count
            simulator_kwargs["dataset_args"]["dag_method"] = "gnp"
            simulator_kwargs["dataset_args"]["gnp_max_n"] = self.task_limit

            # Set Simulator args
            if simulator_mode == "embedded":
                assert "simulator_jar_path" in simulator_kwargs, "simulator_jar_path is required for embedded mode"
                self.simulator = EmbeddedSimulator(**simulator_kwargs)
            elif simulator_mode == "internal":
                self.simulator = InternalSimulator(simulator_kwargs["dataset_args"])
        elif simulator_mode == "remote":
            self.simulator = RemoteSimulator(**simulator_kwargs)
        else:
            raise ValueError(f"Unknown simulator mode: {simulator_mode}")

        # Initialize the renderer
        self.render_mode = env_config.get("render_mode", None)
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        self.renderer = None

    # --------------------- Parse Observation -------------------------------------------------------------------------

    @override
    def parse_obs(self, obs: Any | None) -> dict[str, list[dict[str, Any]]]:
        if isinstance(obs, dict):
            return obs

        if obs is None:
            if self.last_obs is not None:
                return self.last_obs
            return {"tasks": [], "vms": []}
        return {
            "tasks": [
                {
                    "id": int(task.getId()),
                    "workflow_id": int(task.getWorkflowId()),
                    "length": int(task.getLength()),
                    "req_memory_mb": int(task.getReqMemoryMb()),
                    "child_ids": [int(child) for child in task.getChildIds()],
                }
                for task in obs.getTasks()
            ],
            "vms": [
                {
                    "id": int(vm.getId()),
                    "memory_mb": int(vm.getMemoryMb()),
                    "cpu_speed_mips": float(vm.getCpuSpeedMips()),
                    "host_power_idle_watt": float(vm.getHost().getPowerIdleWatt()),
                    "host_power_peak_watt": float(vm.getHost().getPowerPeakWatt()),
                    "host_cpu_speed_mips": float(vm.getHost().getCpuSpeedMips()),
                }
                for vm in obs.getVms()
            ],
        }

    # --------------------- Parse Info --------------------------------------------------------------------------------

    @override
    def parse_info(self, info: Any | None) -> dict[str, Any]:
        if isinstance(info, dict):
            return info

        raw_info = super().parse_info(info)
        parsed_info: dict[str, Any] = {}

        if raw_info.get("solution") is not None:
            solution_json = raw_info["solution"]
            solution_dict = json.loads(solution_json)
            parsed_info["solution"] = Solution.from_json(solution_dict)
        if raw_info.get("total_power_consumption_watt") is not None:
            power_cons_str = raw_info["total_power_consumption_watt"]
            parsed_info["total_power_consumption_watt"] = float(power_cons_str)

        return parsed_info

    # --------------------- Create Action -----------------------------------------------------------------------------

    @override
    def create_action(self, jvm: Any, action: list[dict[str, Any]]) -> Any:
        if jvm is None:
            return action

        assignments = jvm.java.util.ArrayList()
        for act in action:
            assignment = jvm.org.example.api.dtos.VmAssignmentDto(act["vm_id"], act["workflow_id"], act["task_id"])
            assignments.add(assignment)
        return jvm.org.example.api.scheduler.gym.types.StaticAction(assignments)
