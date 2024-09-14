from gymnasium import spaces
from typing import Any, override
import numpy as np

from gym_simulator.releaser.types import ActType, ObsType
from gym_simulator.releaser.renderer import ReleaserHumanRenderer
from gym_simulator.core.environments.cloudsim import BaseCloudSimEnvironment
from gym_simulator.core.simulators.embedded import EmbeddedSimulator
from gym_simulator.core.simulators.remote import RemoteSimulator


class CloudSimReleaserEnvironment(BaseCloudSimEnvironment):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, env_config: dict[str, Any]):
        self.action_space = spaces.Discrete(2)  # 0 - Do nothing, 1 - Release
        self.observation_space = spaces.Tuple(
            [
                spaces.Discrete(1000),  # Buffered tasks
                spaces.Discrete(1000),  # Released tasks
                spaces.Discrete(1000),  # Scheduled tasks
                spaces.Discrete(1000),  # Running tasks
                spaces.Discrete(1000),  # Completed tasks
                spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float64),  # Completion time variance
                spaces.Discrete(1000),  # VM count
            ]
        )

        # Initialize the simulator
        simulator_mode = env_config["simulator_mode"]
        simulator_kwargs = env_config.get("simulator_kwargs", {})
        if simulator_mode == "embedded":
            simulator_kwargs["worker_index"] = getattr(env_config, "worker_index", 0)
            self.simulator = EmbeddedSimulator(**simulator_kwargs)
        elif simulator_mode == "remote":
            self.simulator = RemoteSimulator(**simulator_kwargs)
        else:
            raise ValueError(f"Unknown simulator mode: {simulator_mode}")

        # Initialize the renderer
        render_mode = env_config.get("render_mode", None)
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        if render_mode == "human":
            self.renderer = ReleaserHumanRenderer(self.metadata["render_fps"])

    # --------------------- Parse Observation ---------------------------------------------------------------------------

    @override
    def parse_obs(self, obs: Any | None) -> ObsType:
        if obs is None:
            return self.observation_space.sample()
        return (
            np.int64(obs.bufferedTasks()),
            np.int64(obs.releasedTasks()),
            np.int64(obs.scheduledTasks()),
            np.int64(obs.runningTasks()),
            np.int64(obs.completedTasks()),
            np.array([obs.completionTimeVariance()], dtype=np.float64),
            np.int64(obs.vmCount()),
        )

    # --------------------- Create Action -------------------------------------------------------------------------------

    @override
    def create_action(self, jvm: Any, action: ActType) -> Any:
        return jvm.org.example.api.scheduler.gym.types.ReleaserAction(bool(action == 1))
