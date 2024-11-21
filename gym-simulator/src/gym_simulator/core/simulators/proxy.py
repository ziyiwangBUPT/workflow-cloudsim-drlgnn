import dataclasses
from typing import Any, Callable
from gym_simulator.core.simulators.base import BaseSimulator
from gym_simulator.core.types import TaskDto


class InternalProxySimulator(BaseSimulator):
    running = False

    def __init__(self, obs: "InternalProxySimulatorObs") -> None:
        self.obs = obs

    def start(self):
        assert not self.is_running()
        self.running = True

    def stop(self) -> str | None:
        assert self.is_running()
        self.running = False
        return None

    def is_running(self) -> bool:
        return self.running

    def reset(self, seed: int | None) -> Any:
        if self.is_running():
            self.stop()
        self.start()
        return self.obs.getObservation()

    def step(self, action_creator: Callable[[Any], Any]) -> Any:
        assert self.is_running()
        action = action_creator(None)
        return self.obs

    def reboot(self):
        if self.is_running():
            self.stop()
        self.start()


class InternalProxySimulatorObs:
    tasks: list[TaskDto] = []
    vms: list[TaskDto] = []

    def getObservation(self):
        return {
            "tasks": [dataclasses.asdict(task) for task in self.tasks],
            "vms": [dataclasses.asdict(vm) for vm in self.vms],
        }

    def getReward(self):
        return 0

    def isTerminated(self):
        return True

    def isTruncated(self):
        return False

    def getInfo(self):
        return {}
