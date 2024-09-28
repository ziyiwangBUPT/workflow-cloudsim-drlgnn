from abc import ABC

from gym_simulator.algorithms.base_heuristic import TaskIdType
from gym_simulator.algorithms.random_min import RandomMinScheduler


class MinMinScheduler(RandomMinScheduler, ABC):
    def choose_next(self, ready_tasks: list[TaskIdType]) -> TaskIdType:
        smallest_task_id = None
        smallest_task_length = float("inf")
        for task_id in ready_tasks:
            task = self.get_task(task_id)
            if task.length < smallest_task_length:
                smallest_task_length = task.length
                smallest_task_id = self.tid(task)
        assert smallest_task_id is not None

        return smallest_task_id
