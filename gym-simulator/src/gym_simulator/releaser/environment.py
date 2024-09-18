import socket

from gymnasium import spaces
from typing import Any, override
import numpy as np

from gym_simulator.releaser.types import ActType, ObsType
from gym_simulator.releaser.renderer import ReleaserRenderer
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


class CloudSimReleaserEnvironment(BaseCloudSimEnvironment):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, env_config: dict[str, Any]):
        # 0 - Do nothing, 1 - Release
        self.action_space = spaces.Discrete(2)
        # Buffered tasks, released tasks, scheduled tasks, running tasks, completed tasks, completion time variance, VM count
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32,
        )

        # Initialize the simulator
        simulator_mode = env_config["simulator_mode"]
        simulator_kwargs = env_config.get("simulator_kwargs", {})
        if simulator_mode == "embedded":
            simulator_kwargs["jvm_port"] = free_port()
            self.simulator = EmbeddedSimulator(**simulator_kwargs)
        elif simulator_mode == "remote":
            self.simulator = RemoteSimulator(**simulator_kwargs)
        else:
            raise ValueError(f"Unknown simulator mode: {simulator_mode}")

        # Initialize the renderer
        self.render_mode = env_config.get("render_mode", None)
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        self.renderer = ReleaserRenderer(self.metadata["render_fps"])

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # --- Test ---
        assert self.last_obs is not None
        x = self.last_obs[0] - self.last_obs[1]
        reward1 = int(90 <= x <= 110) if action == 1 else 0
        reward2 = 1 - abs(x - 100) / 100
        reward = (reward1 + reward2) / 2
        # --- Test ---

        obs, _, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info

    # --------------------- Parse Observation ---------------------------------------------------------------------------

    @override
    def parse_obs(self, obs: Any | None) -> ObsType:
        if obs is None:
            if self.last_obs is not None:
                return self.last_obs
            return np.zeros(7, dtype=np.float32)
        return np.array(
            [
                obs.bufferedTasks(),
                obs.releasedTasks(),
                obs.scheduledTasks(),
                obs.runningTasks(),
                obs.completedTasks(),
                obs.completionTimeVariance(),
                obs.vmCount(),
            ],
            dtype=np.float32,
        )

    # --------------------- Create Action -------------------------------------------------------------------------------

    @override
    def create_action(self, jvm: Any, action: ActType) -> Any:
        return jvm.org.example.api.scheduler.gym.types.ReleaserAction(bool(action == 1))
