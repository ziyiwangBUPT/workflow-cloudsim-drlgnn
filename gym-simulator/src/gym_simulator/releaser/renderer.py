import matplotlib.pyplot as plt

from typing import override
from gym_simulator.releaser.types import ObsType
from gym_simulator.core.renderers.pygame import PygameRenderer


class ReleaserRenderer(PygameRenderer):
    """
    A human renderer for the Releaser environment.
    Uses Pygame to render the environment.
    """

    def __init__(self, render_fps: int):
        super().__init__(render_fps, 800, 600)

    @override
    def draw_chart(self, obs: ObsType):
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

        return fig
