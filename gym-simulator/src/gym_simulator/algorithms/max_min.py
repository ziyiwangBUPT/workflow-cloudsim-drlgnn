from typing import override

from gym_simulator.algorithms.min_min import MinMinScheduler
from gym_simulator.algorithms.types import TaskDto


class MaxMinScheduler(MinMinScheduler):
    """
    Implementation of the MaxMin scheduling algorithm.

    MaxMin is a simple scheduling algorithm that schedules the task with the largest length
    on the VM that will complete the task the fastest.
    This is a variant of the MinMin algorithm choosing the task with the largest length first.
    """

    @override
    def select_task(self, ready_tasks: list[TaskDto]) -> TaskDto:
        """Choose the task with the largest length."""
        largest_task = None
        largest_task_length = -float("inf")
        for task in ready_tasks:
            if task.length > largest_task_length:
                largest_task_length = task.length
                largest_task = task
        assert largest_task is not None

        return largest_task
