import io

import matplotlib.pyplot as plt
import pygraphviz as pgv


def draw_agraph(ax: plt.Axes, A: pgv.AGraph):
    """
    Draw the provided AGraph on the provided Axes.
    """

    A.layout(prog="dot")
    buffer = io.BytesIO()
    buffer.write(A.draw(format="png"))
    buffer.seek(0)
    ax.imshow(plt.imread(buffer))
    ax.axis("off")


def save_agraph(A: pgv.AGraph, path: str, dir_lr: bool = False):
    """
    Save the provided AGraph to the provided path.
    """

    A.layout(prog="dot", args="-Grankdir=LR" if dir_lr else "")
    A.draw(path)
