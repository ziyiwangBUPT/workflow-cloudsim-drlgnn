import gymnasium as gym
from gymnasium import spaces
from typing import Any

from py4j.java_gateway import JavaGateway

from gym_simulator.releaser.types import ActionType, ObsType
from gym_simulator.releaser.renderer import ReleaserRenderer, ReleaserPlotRenderer


class CloudSimReleaserEnv(gym.Env):
    action_space: spaces.Discrete
    observation_space: spaces.Tuple
    _gateway: JavaGateway

    _connector: Any
    _last_observation: ObsType | None
    _renderer: ReleaserRenderer

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(2)  # 0 - Do nothing, 1 - Release
        self.observation_space = spaces.Tuple(
            [
                spaces.Discrete(1000),  # Buffered tasks
                spaces.Discrete(1000),  # Released tasks
                spaces.Discrete(1000),  # Scheduled tasks
                spaces.Discrete(1000),  # Running tasks
                spaces.Discrete(1000),  # Completed tasks
                spaces.Discrete(1000),  # VM count
            ]
        )

        # Initialize the Java Gateway
        self._gateway = JavaGateway()
        self._connector = self._gateway.entry_point

        # Renderer
        self._last_observation = None
        self._renderer = ReleaserPlotRenderer()

    def _parse_obs(self, observation: Any) -> ObsType:
        if observation is None:
            return self._last_observation or (0, 0, 0, 0, 0, 0)
        return (
            int(observation.bufferedTasks()),
            int(observation.releasedTasks()),
            int(observation.scheduledTasks()),
            int(observation.runningTasks()),
            int(observation.completedTasks()),
            int(observation.vmCount()),
        )

    def step(self, action: ActionType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        # Step the environment
        action_obj = self._gateway.jvm.org.example.api.scheduler.gym.types.ReleaserAction(bool(action == 1))
        result = self._connector.step(action_obj)

        # Parse the result
        observation = self._parse_obs(result.getObservation())
        reward = float(result.getReward())
        terminated = bool(result.isTerminated())
        truncated = bool(result.isTruncated())
        info: dict[str, Any] = {}

        self._last_observation = observation
        self.render()

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # Reset the environment
        result = self._connector.reset()

        # Parse the result
        observation = self._parse_obs(result)
        info: dict[str, Any] = {}

        self.render()

        return observation, info

    def render(self):
        if self._last_observation is not None:
            self._renderer.update(self._last_observation)

    def close(self):
        self._gateway.shutdown()
