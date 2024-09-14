import io

from abc import ABC, abstractmethod
from typing import Any, override
import pygame

from gym_simulator.core.renderers.base import BaseRenderer


class PygameRenderer(BaseRenderer, ABC):
    """
    Abstract base class for creating a renderer using Pygame for visualizing the environment's state.

    This class defines the standard methods for updating the renderer with new observations and closing it.
    The rendering is done by drawing the observation as an image and displaying it on the screen.
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

    @override
    def update(self, obs: Any):
        # Lazy initialization
        if self.window is None or self.clock is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        # Draw the observation
        buf = self.draw(obs)
        self.window.blit(pygame.image.load(buf), (0, 0))

        # Update the display
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.render_fps)

    # --------------------- Renderer Close ---------------------------------------------------------------------------

    @override
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    # --------------------- Draw -------------------------------------------------------------------------------------

    @abstractmethod
    def draw(self, obs: Any) -> io.BytesIO:
        """Draw the observation as an image and return the image buffer"""
        raise NotImplementedError
