import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from dataset_generator.visualizers.plotters import plot_2d_matrix
from dataset_generator.visualizers.utils import draw_agraph
from gym_simulator.core.renderers.pygame import PygameRenderer
from gym_simulator.environments.states.rl import RlEnvState


class RlEnvironmentRenderer(PygameRenderer):
    def draw_chart(self, state: RlEnvState) -> Figure:
        fig, axes = plt.subplots(
            ncols=2,
            nrows=2,
            figsize=(self.width / 100, self.height / 100),
            gridspec_kw={"width_ratios": [1, 2], "height_ratios": [9, 1]},
            dpi=100,
        )
        plot_2d_matrix(axes[0][0], "Task-VM Compatibility", state.task_vm_time_cost)

        G: nx.DiGraph = nx.DiGraph()
        for i in range(len(state.task_state_scheduled)):
            if state.task_state_scheduled[i] == 1:
                assert state.task_state_ready[i] == 0
                label = f"{i} ({state.assignments[i]})\n{state.task_completion_time[i]:.1f}"
                G.add_node(i, color="green", label=label)
            elif state.task_state_ready[i] == 1:
                label = f"{i}\n{state.task_completion_time[i]:.1f}"
                G.add_node(i, color="red", label=label)
            else:
                label = f"{i}\n{state.task_completion_time[i]:.1f}"
                G.add_node(i, color="black", label=label)

        for i in range(len(state.task_graph_edges)):
            for j in range(len(state.task_graph_edges[i])):
                if state.task_graph_edges[i][j]:
                    G.add_edge(i, j, color="black")

        A = nx.nx_agraph.to_agraph(G)
        draw_agraph(axes[0][1], A)

        vm_state = np.vstack([state.vm_completion_time])
        plot_2d_matrix(axes[1][0], "VM State", vm_state)

        assignments = np.vstack([state.assignments])
        plot_2d_matrix(axes[1][1], "Assignments", assignments)

        fig.tight_layout()
        return fig
