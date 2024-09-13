import io
import abc
import pygame

import matplotlib.pyplot as plt


from gym_simulator.releaser.types import ObsType


class ReleaserRenderer(abc.ABC):
    @abc.abstractmethod
    def update(self, obs: tuple[int, int, int, int, int, int]):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError


class ReleaserPlotRenderer(ReleaserRenderer):
    width: int = 800
    height: int = 600
    render_fps: int
    _window: pygame.Surface | None
    _clock: pygame.time.Clock | None

    def __init__(self, render_fps: int):
        self._window = None
        self._clock = None
        self.render_fps = render_fps

    def update(self, obs: ObsType):
        """Update the renderer with the given observation"""
        if self._window is None:
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode((self.width, self.height))
        if self._clock is None:
            self._clock = pygame.time.Clock()

        buf = self._draw(obs)
        self._window.blit(pygame.image.load(buf), (0, 0))

        pygame.event.pump()
        pygame.display.update()
        self._clock.tick(self.render_fps)

    def close(self):
        """Close the renderer"""
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()

    def _draw(self, obs: ObsType) -> io.BytesIO:
        """Draw the observation on the screen"""
        buffered, released, scheduled, executed, completed, _, __ = obs
        inBuffering = buffered - released
        inReleasing = released - scheduled
        inScheduling = scheduled - executed
        inExecuting = executed - completed
        inCompleted = completed

        fig, ax = plt.subplots()
        ax.barh(
            ["Buffer", "Released", "Ready", "Running", "Completed"],
            [inBuffering, inReleasing, inScheduling, inExecuting, inCompleted],
            color=["blue", "green", "red", "yellow", "purple"],
        )
        ax.set_xlim(0, 1000)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)

        return buf
