import gymnasium as gym
from gymnasium import spaces
from typing import Any

from py4j.java_gateway import JavaGateway

from gym_simulator.releaser.types import ActionType, ObsType
from gym_simulator.releaser.renderer import ReleaserRenderer, ReleaserPlotRenderer
from gym_simulator.core.runner import NoOpSimulatorRunner, SimulatorRunner


class CloudSimReleaserEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    action_space: spaces.Discrete
    observation_space: spaces.Tuple
    render_mode: str

    _gateway: JavaGateway
    _connector: Any
    _last_observation: ObsType | None = None
    _human_renderer: ReleaserRenderer
    _runner: SimulatorRunner

    # --------------------- Initialization ------------------------------------

    def __init__(self, runner: SimulatorRunner | None = None, render_mode=None):
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
        self._runner = runner or NoOpSimulatorRunner()
        self._gateway = JavaGateway()
        self._connector = self._gateway.entry_point

        # Set the render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._human_renderer = ReleaserPlotRenderer(self.metadata["render_fps"])

    # --------------------- Reset ---------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        # Restart the simulator
        self._stop_simulator()
        self._runner.run()

        # Get the initial observation
        result = self._connector.reset()
        observation = self._parse_obs(result)
        info: dict[str, Any] = {}

        # Render the frame
        self._last_observation = observation
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    # --------------------- Step ----------------------------------------------

    def step(self, action: ActionType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        # Step the environment
        action_obj = self._create_action(action)
        result = self._connector.step(action_obj)
        observation = self._parse_obs(result.getObservation())
        reward = float(result.getReward())
        terminated = bool(result.isTerminated())
        truncated = bool(result.isTruncated())
        info: dict[str, Any] = {}

        # Render the frame
        self._last_observation = observation
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    # --------------------- Rendering -----------------------------------------

    def render(self):
        # Rendering is hadled by the environment
        pass

    # --------------------- Close ----------------------------------------------

    def close(self):
        self._stop_simulator()
        self._human_renderer.close()

    # --------------------- Private methods -----------------------------------

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

    def _create_action(self, action: ActionType) -> Any:
        return self._gateway.jvm.org.example.api.scheduler.gym.types.ReleaserAction(bool(action == 1))

    def _render_frame(self):
        assert self._last_observation is not None
        self._human_renderer.update(self._last_observation)

    def _stop_simulator(self):
        if self._runner.is_running():
            self._runner.stop()
        self._gateway.close()
