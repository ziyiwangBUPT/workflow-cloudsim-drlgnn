from abc import abstractmethod

import gymnasium as gym
from typing import Any, Generic, TypeVar, override

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
    renderer: BaseRenderer | None = None

    # --------------------- Reset --------------------------------------------------------------------------------------

    @override
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        # Send the reset command to the simulator
        result = self.simulator.reset()
        obs = self.parse_obs(result)
        info: dict[str, Any] = {}

        # Update the renderer
        if self.renderer is not None:
            self.renderer.update(obs)

        return obs, info

    # --------------------- Step ---------------------------------------------------------------------------------------

    @override
    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        # Send the action to the simulator
        result = self.simulator.step(lambda jvm: self.create_action(jvm, action))
        obs = self.parse_obs(result.getObservation())
        reward = float(result.getReward())
        terminated = bool(result.isTerminated())
        truncated = bool(result.isTruncated())
        info: dict[str, Any] = {}

        if terminated or truncated:
            self.simulator.stop()

        # Update the renderer
        if not terminated and not truncated:
            if self.renderer is not None:
                self.renderer.update(obs)

        return obs, reward, terminated, truncated, info

    # --------------------- Close --------------------------------------------------------------------------------------

    @override
    def close(self):
        self.simulator.stop()
        if self.renderer is not None:
            self.renderer.close()

    # --------------------- Destructor ---------------------------------------------------------------------------------

    @override
    def __del__(self):
        self.close()

    # --------------------- Abstract Methods ---------------------------------------------------------------------------

    @abstractmethod
    def parse_obs(self, obs: Any | None) -> ObsType:
        """Parse the raw observation object from the simulator into the observation space of the environment."""
        raise NotImplementedError

    @abstractmethod
    def create_action(self, jvm: Any, action: ActType) -> Any:
        """Create an action object to be sent to the simulator based on the action space of the environment."""
        raise NotImplementedError
