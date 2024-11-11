from abc import abstractmethod

import gymnasium as gym
from typing import Any, Generic, TypeVar

from gym_simulator.core.simulators.base import BaseSimulator
from gym_simulator.core.renderers.base import BaseRenderer


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseCloudSimEnvironment(gym.Env, Generic[ObsType, ActType]):
    """
    Abstract base class for creating a cloud simulation environment within the Gym framework.

    This environment integrates a simulator for the cloud systems' dynamics and optionally a renderer
    for visualizations. It defines standard methods for environment interaction like reset and step,
    following OpenAI Gym's interface.
    """

    simulator: BaseSimulator
    renderer: BaseRenderer | None
    last_obs: ObsType | None = None

    # --------------------- Reset --------------------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        # Send the reset command to the simulator
        result = self.simulator.reset(seed)
        obs = self.parse_obs(result)
        info: dict[str, Any] = {}

        # Update the renderer
        self.last_obs = obs
        if self.render_mode == "human":
            self.render()

        return obs, info

    # --------------------- Step ---------------------------------------------------------------------------------------

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        # Send the action to the simulator
        result = self.simulator.step(lambda jvm: self.create_action(jvm, action))
        obs = self.parse_obs(result.getObservation())
        reward = float(result.getReward())
        terminated = bool(result.isTerminated())
        truncated = bool(result.isTruncated())
        info: dict[str, Any] = self.parse_info(result.getInfo())

        if terminated or truncated:
            output = self.simulator.stop()
            info["stdout"] = output

        # Update the renderer
        if not terminated and not truncated:
            self.last_obs = obs
            if self.render_mode == "human":
                self.render()

        return obs, reward, terminated, truncated, info

    # --------------------- Render--------------------------------------------------------------------------------------

    def render(self):
        if self.last_obs is None:
            return None

        if self.render_mode == "human":
            assert self.renderer is not None, "Human rendering requires a renderer"
            self.renderer.update(self.last_obs)
        elif self.render_mode == "rgb_array":
            assert self.renderer is not None, "RGB array rendering requires a renderer"
            return self.renderer.draw(self.last_obs)
        return None

    # --------------------- Close --------------------------------------------------------------------------------------

    def close(self):
        if not hasattr(self, "simulator"):
            return
        if self.simulator.is_running():
            self.simulator.stop()
        if self.renderer is not None:
            self.renderer.close()

    # --------------------- Destructor ---------------------------------------------------------------------------------

    def __del__(self):
        self.close()

    # --------------------- Abstract Methods ---------------------------------------------------------------------------

    @abstractmethod
    def parse_obs(self, obs: Any | None) -> ObsType:
        """Parse the raw observation object from the simulator into the observation space of the environment."""
        raise NotImplementedError

    def parse_info(self, info: Any) -> dict[str, Any]:
        """Parse the raw info object from the simulator into the info space of the environment."""
        return {str(o.getKey()): str(o.getValue()) for o in info.entrySet()}

    @abstractmethod
    def create_action(self, jvm: Any, action: ActType) -> Any:
        """Create an action object to be sent to the simulator based on the action space of the environment."""
        raise NotImplementedError
