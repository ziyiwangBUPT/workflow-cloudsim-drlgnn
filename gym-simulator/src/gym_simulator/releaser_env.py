import json

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any
import matplotlib.pyplot as plt

from py4j.java_gateway import JavaGateway


class CloudSimReleaserEnv(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self):
        super().__init__()

        # Actions: 0 - Do nothing, 1 - Release
        self.action_space = spaces.Discrete(2)
        # States: (Cached Jobs, VM count, Completion time variance)
        self.observation_space = spaces.Tuple(
            [
                spaces.Discrete(1000),
                spaces.Discrete(1000),
                spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            ]
        )

        # Initialize the Java Gateway
        self.gateway = JavaGateway()
        self.connector = self.gateway.entry_point

        self.job_counts = []
        plt.ion()
        self.fig, self.ax = plt.subplots()
        (self.jobs_line,) = self.ax.plot([], [])
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Count")

    def step(self, action: int) -> tuple[tuple[int, int, float], float, bool, bool, dict[str, Any]]:
        # Convert the action to a Java object
        action_obj = self.gateway.jvm.org.example.api.scheduler.gym.types.ReleaserAction(bool(action == 1))
        # Step the environment
        result_str = self.connector.step(action_obj)

        # Parse the result
        result = json.loads(result_str)
        observation_dict: dict[str, Any] = result.get("observation", {})
        observation = (
            int(observation_dict.get("cached_jobs", 0)),
            int(observation_dict.get("vm_count", 0)),
            float(observation_dict.get("completion_time_variance", 0)),
        )
        reward = float(result.get("reward", 0))
        terminated = bool(result.get("terminated", False))
        truncated = bool(result.get("truncated", False))
        info: dict[str, Any] = {}

        self.job_counts.append(observation[0])
        self.render()

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # Reset the environment
        result_str = self.connector.reset()

        # Parse the result
        result = json.loads(result_str)
        observation_dict: dict[str, Any] = result or {}
        observation = (
            int(observation_dict.get("cached_jobs", 0)),
            int(observation_dict.get("vm_count", 0)),
            float(observation_dict.get("completion_time_variance", 0)),
        )
        info: dict[str, Any] = {}

        self.job_counts.append(observation[0])
        self.render()

        return observation, info

    def render(self):
        self.jobs_line.set_xdata(range(len(self.job_counts)))
        self.jobs_line.set_ydata(self.job_counts)
        self.ax.set_xlim(0, len(self.job_counts))
        self.ax.set_ylim(0, max(self.job_counts))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def close(self):
        self.gateway.shutdown()
        plt.pause(10)
        plt.show()
