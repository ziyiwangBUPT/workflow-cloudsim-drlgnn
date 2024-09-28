from gym_simulator.algorithms.min_min import MinMinScheduler
from gym_simulator.algorithms.types import TaskIdType


class MaxMinScheduler(MinMinScheduler):
    """
    Implementation of the MaxMin scheduling algorithm.

    MaxMin is a simple scheduling algorithm that schedules the task with the largest length
    on the VM that will complete the task the fastest.
    This is a variant of the MinMin algorithm choosing the task with the largest length first.
    """

    def choose_next(self, ready_tasks: list[TaskIdType]) -> TaskIdType:
        """Choose the task with the largest length."""
        largest_task_id = None
        largest_task_length = -float("inf")
        for task_id in ready_tasks:
            task = self.get_task(task_id)
            if task.length > largest_task_length:
                largest_task_length = task.length
                largest_task_id = self.tid(task)
        assert largest_task_id is not None

        return largest_task_id
