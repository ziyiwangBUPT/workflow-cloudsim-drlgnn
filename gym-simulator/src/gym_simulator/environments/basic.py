import json

from gymnasium import spaces
from typing import Any
import numpy as np

from dataset_generator.core.models import Solution
from gym_simulator.core.environments.cloudsim import BaseCloudSimEnvironment
from gym_simulator.core.simulators.embedded import EmbeddedSimulator
from gym_simulator.core.simulators.internal import InternalSimulator
from gym_simulator.core.simulators.proxy import InternalProxySimulator
from gym_simulator.core.simulators.remote import RemoteSimulator


class BasicCloudSimEnvironment(BaseCloudSimEnvironment):
    metadata = {"render_modes": []}

    def __init__(self, env_config: dict[str, Any]):
        self.action_space = spaces.Sequence(
            spaces.Dict(
                {
                    "vm_id": spaces.Discrete(42),
                    "workflow_id": spaces.Discrete(42),
                    "task_id": spaces.Discrete(42),
                }
            )
        )
        self.observation_space = spaces.Dict(
            {
                "tasks": spaces.Sequence(
                    spaces.Dict(
                        {
                            "id": spaces.Discrete(42),
                            "workflow_id": spaces.Discrete(42),
                            "length": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int64),
                            "req_memory_mb": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int64),
                            "child_ids": spaces.Sequence(spaces.Discrete(42)),
                        }
                    )
                ),
                "vms": spaces.Sequence(
                    spaces.Dict(
                        {
                            "id": spaces.Discrete(42),
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
            # Set Simulator args
            if simulator_mode == "embedded":
                self.simulator = EmbeddedSimulator(
                    simulator_jar_path=simulator_kwargs["simulator_jar_path"],
                    scheduler_preset=simulator_kwargs["scheduler_preset"],
                    dataset_args=simulator_kwargs.get("dataset_args", {}),
                    remote_debug=simulator_kwargs.get("remote_debug", False),
                    verbose=simulator_kwargs.get("verbose", False),
                )
            elif simulator_mode == "internal":
                self.simulator = InternalSimulator(dataset_args=simulator_kwargs["dataset_args"])
        elif simulator_mode == "proxy":
            self.simulator = InternalProxySimulator(obs=simulator_kwargs["proxy_obs"])
        elif simulator_mode == "remote":
            self.simulator = RemoteSimulator()
        else:
            raise ValueError(f"Unknown simulator mode: {simulator_mode}")

        # Initialize the renderer
        self.render_mode = env_config.get("render_mode", None)
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        self.renderer = None

    # --------------------- Parse Observation -------------------------------------------------------------------------

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

    def parse_info(self, info: Any | None) -> dict[str, Any]:
        if isinstance(info, dict):
            return info

        raw_info = super().parse_info(info)
        parsed_info: dict[str, Any] = {}

        if raw_info.get("solution") is not None:
            solution_json = raw_info["solution"]
            solution_dict = json.loads(solution_json)
            parsed_info["solution"] = Solution.from_json(solution_dict)
        if raw_info.get("total_energy_consumption_j") is not None:
            power_cons_str = raw_info["total_energy_consumption_j"]
            parsed_info["total_energy_consumption_j"] = float(power_cons_str)

        return parsed_info

    # --------------------- Create Action -----------------------------------------------------------------------------

    def create_action(self, jvm: Any, action: list[dict[str, Any]]) -> Any:
        if jvm is None:
            return action

        assignments = jvm.java.util.ArrayList()
        for act in action:
            assignment = jvm.org.example.api.dtos.VmAssignmentDto(act["vm_id"], act["workflow_id"], act["task_id"])
            assignments.add(assignment)
        return jvm.org.example.api.scheduler.gym.types.StaticAction(assignments)
