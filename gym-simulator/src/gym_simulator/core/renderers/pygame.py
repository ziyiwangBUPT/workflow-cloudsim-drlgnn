import io

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy.typing as npt
import numpy as np
from abc import ABC, abstractmethod
import pygame

from gym_simulator.core.renderers.base import BaseRenderer


class PygameRenderer(BaseRenderer, ABC):
    """
    Abstract base class for creating a renderer using Pygame for visualizing the environment's state.

    This class defines the standard methods for updating the renderer with new observations and closing it.
    The rendering is done by drawing the observation as an matplotlib chart and displaying it on the screen.
    """

    window: pygame.Surface | None
    clock: pygame.time.Clock | None

    def __init__(self, render_fps: int, width: int = 800, height: int = 600):
        self.render_fps = render_fps
        self.width = width
        self.height = height
        self.window = None
        self.clock = None

    # --------------------- Renderer Update ---------------------------------------------------------------------------

    def update(self, obs: Any):
        # Lazy initialization
        if self.window is None or self.clock is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        # Draw the observation
        arr = self.draw(obs)
        self.window.blit(pygame.surfarray.make_surface(arr.swapaxes(0, 1)), (0, 0))

        # Update the display
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.render_fps)

    # --------------------- Renderer Close ---------------------------------------------------------------------------

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    # --------------------- Draw Chart -------------------------------------------------------------------------------

    def draw(self, obs: Any) -> npt.NDArray[np.uint8]:
        fig = self.draw_chart(obs)

        # Save the figure as an image buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="raw")
        plt.close(fig)
        buf.seek(0)

        # Convert the image buffer to a numpy array
        arr = np.reshape(
            np.frombuffer(buf.getvalue(), dtype=np.uint8),
            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
        )
        # Drop alpha channel
        arr = arr[:, :, :3]
        buf.close()

        return arr

    @abstractmethod
    def draw_chart(self, obs: Any) -> Figure:
        """Draw the observation as an image and return the image buffer"""
        raise NotImplementedError
