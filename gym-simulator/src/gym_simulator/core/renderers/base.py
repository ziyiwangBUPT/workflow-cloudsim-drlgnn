from abc import ABC, abstractmethod

import numpy.typing as npt
import numpy as np
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

    @abstractmethod
    def draw(self, obs: Any) -> npt.NDArray[np.uint8]:
        """Draw the observation as an image and return the image buffer."""
        raise NotImplementedError
