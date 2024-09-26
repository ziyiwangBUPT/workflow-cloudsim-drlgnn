import socket

from gymnasium import spaces
from typing import Any, override, Generic
import numpy as np

from gym_simulator.core.environments.cloudsim import BaseCloudSimEnvironment
from gym_simulator.core.simulators.embedded import EmbeddedSimulator
from gym_simulator.core.simulators.remote import RemoteSimulator


# Taken from Selenium's utils.py
# https://github.com/SeleniumHQ/selenium/blob/35dd34afbdd96502066d0f7b6a2460a11e5fb73a/py/selenium/webdriver/common/utils.py#L31
def free_port() -> int:
    """Determines a free port using sockets."""
    free_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free_socket.bind(("127.0.0.1", 0))
    free_socket.listen(5)
    port: int = free_socket.getsockname()[1]
    free_socket.close()
    return port


class BasicCloudSimEnvironment(BaseCloudSimEnvironment):
    metadata = {"render_modes": []}

    def __init__(self, env_config: dict[str, Any]):
        host_count = env_config["host_count"]
        vm_count = env_config["vm_count"]
        workflow_count = env_config["workflow_count"]
        task_limit = env_config["task_limit"]

        self.action_space = spaces.Sequence(
            spaces.Dict(
                {
                    "vm_id": spaces.Discrete(vm_count),
                    "workflow_id": spaces.Discrete(workflow_count),
                    "task_id": spaces.Discrete(task_limit),
                }
            )
        )
        self.observation_space = spaces.Dict(
            {
                "tasks": spaces.Sequence(
                    spaces.Dict(
                        {
                            "id": spaces.Discrete(task_limit),
                            "workflow_id": spaces.Discrete(workflow_count),
                            "length": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int64),
                            "req_cores": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int64),
                            "child_ids": spaces.Sequence(spaces.Discrete(task_limit)),
                        }
                    )
                ),
                "vms": spaces.Sequence(
                    spaces.Dict(
                        {
                            "id": spaces.Discrete(vm_count),
                            "cores": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int64),
                            "cpu_speed_mips": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
                            "host_power_idle_watt": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
                            "host_power_peak_watt": spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
                        }
                    )
                ),
            }
        )

        # Initialize the simulator
        simulator_mode = env_config["simulator_mode"]
        simulator_kwargs = env_config.get("simulator_kwargs", {})
        if simulator_mode == "embedded":
            # Set dataset args
            simulator_kwargs["dataset_args"] = simulator_kwargs.get("dataset_args", {})
            assert "host_count" not in simulator_kwargs["dataset_args"], "host_count is set by the environment"
            assert "vm_count" not in simulator_kwargs["dataset_args"], "vm_count is set by the environment"
            assert "workflow_count" not in simulator_kwargs["dataset_args"], "workflow_count is set by the environment"
            assert "dag_method" not in simulator_kwargs["dataset_args"], "dag_method is set by the environment"
            assert "gnp_max_n" not in simulator_kwargs["dataset_args"], "gnp_max_n is set by the environment"
            simulator_kwargs["dataset_args"]["host_count"] = host_count
            simulator_kwargs["dataset_args"]["vm_count"] = vm_count
            simulator_kwargs["dataset_args"]["workflow_count"] = workflow_count
            simulator_kwargs["dataset_args"]["dag_method"] = "gnp"
            simulator_kwargs["dataset_args"]["gnp_max_n"] = task_limit

            # Set Simulator args
            assert "jvm_port" not in simulator_kwargs, "jvm_port is set by the environment"
            assert "simulator_jar_path" in simulator_kwargs, "simulator_jar_path is required for embedded mode"
            simulator_kwargs["jvm_port"] = free_port()
            self.simulator = EmbeddedSimulator(**simulator_kwargs)
        elif simulator_mode == "remote":
            self.simulator = RemoteSimulator(**simulator_kwargs)
        else:
            raise ValueError(f"Unknown simulator mode: {simulator_mode}")

        # Initialize the renderer
        self.render_mode = env_config.get("render_mode", None)
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        self.renderer = None

    # --------------------- Parse Observation ---------------------------------------------------------------------------

    @override
    def parse_obs(self, obs: Any | None) -> dict[str, list[dict[str, Any]]]:
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
                    "req_cores": int(task.getReqCores()),
                    "child_ids": [int(child) for child in task.getChildIds()],
                }
                for task in obs.getTasks()
            ],
            "vms": [
                {
                    "id": int(vm.getId()),
                    "cores": int(vm.getCores()),
                    "cpu_speed_mips": float(vm.getCpuSpeedMips()),
                    "host_power_idle_watt": float(vm.getHost().getPowerIdleWatt()),
                    "host_power_peak_watt": float(vm.getHost().getPowerPeakWatt()),
                }
                for vm in obs.getVms()
            ],
        }

    # --------------------- Create Action -------------------------------------------------------------------------------

    @override
    def create_action(self, jvm: Any, action: list[dict[str, Any]]) -> Any:
        assignments = jvm.java.util.ArrayList()
        for act in action:
            assignment = jvm.org.example.api.dtos.VmAssignmentDto(act["vm_id"], act["workflow_id"], act["task_id"])
            assignments.add(assignment)
        return jvm.org.example.api.scheduler.gym.types.StaticAction(assignments)
