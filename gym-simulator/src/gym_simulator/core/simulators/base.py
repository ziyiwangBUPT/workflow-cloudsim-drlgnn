from abc import ABC, abstractmethod

from io import UnsupportedOperation
from typing import Any, Callable


class BaseSimulator(ABC):
    """
    Abstract base class for creating a simulator for running the environment's dynamics.

    This class defines the standard methods for starting and stopping the simulator, resetting the environment, and
    stepping the environment dynamics with an action.
    """

    @abstractmethod
    def start(self):
        """Start the simulator."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> str | None:
        """Stop the simulator and (optionally) return the STDOUT output."""
        raise NotImplementedError

    @abstractmethod
    def is_running(self) -> bool:
        """Return whether the simulator is running."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, seed: int) -> Any:
        """Reset the environment and return the initial observation."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action_creator: Callable[[Any], Any]) -> Any:
        """Step the environment dynamics with the given action creator and return the new observation."""
        raise NotImplementedError

    def reboot(self):
        """Reboot the simulator. Not all simulators support this operation."""
        raise UnsupportedOperation("Reboot operation is not supported by this simulator.")
