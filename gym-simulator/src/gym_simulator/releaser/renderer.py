import io
import abc
import pygame

import matplotlib.pyplot as plt


from gym_simulator.releaser.types import ObsType


class ReleaserRenderer(abc.ABC):
    @abc.abstractmethod
    def update(self, obs: tuple[int, int, int, int, int, int]):
        raise NotImplementedError


class ReleaserPlotRenderer(ReleaserRenderer):
    width: int = 800
    height: int = 600

    def __init__(self):
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

    def update(self, obs: ObsType):
        buffered, released, scheduled, executed, completed, vm_count = obs
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

        self.screen.blit(pygame.image.load(buf), (0, 0))
        pygame.display.flip()
        self.clock.tick(10)
