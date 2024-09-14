from abc import ABC, abstractmethod


from typing import Any


class BaseRenderer(ABC):
    """
    Abstract base class for creating a renderer for visualizing the environment's state.

    This class defines the standard methods for updating the renderer with new observations and closing it.
    """

    @abstractmethod
    def update(self, obs: Any):
        """Update the renderer with the new observation."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Close the renderer."""
        raise NotImplementedError
