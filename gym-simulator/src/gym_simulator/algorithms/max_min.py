from abc import ABC

from gym_simulator.algorithms.base_ready_queue import TaskIdType
from gym_simulator.algorithms.random_min import RandomMinScheduler


class MaxMinScheduler(RandomMinScheduler, ABC):
    def choose_next(self, ready_tasks: list[TaskIdType]) -> TaskIdType:
        largest_task_id = None
        largest_task_length = -float("inf")
        for task_id in ready_tasks:
            task = self.get_task(task_id)
            if task.length > largest_task_length:
                largest_task_length = task.length
                largest_task_id = self.tid(task)
        assert largest_task_id is not None

        return largest_task_id
