import time

from typing import Any, Callable, override
from gym_simulator.core.simulators.base import BaseSimulator
from py4j.java_gateway import JavaGateway


class RemoteSimulator(BaseSimulator):
    """
    A simulator that connects to a remote CloudSim simulator using Py4J.
    This simulator will not start or stop the simulator process, but will connect to an existing simulator process.

    The simulator communicates with the Java process using the Py4J library, which allows calling Java methods
    from Python and vice versa. The simulator uses a Java gateway to connect to the Java process and send commands.
    """

    java_gateway: JavaGateway

    def __init__(self):
        self.java_gateway = JavaGateway()
        self.env_connector = self.java_gateway.entry_point

    # --------------------- Simulator Control -------------------------------------------------------------------------

    @override
    def start(self):
        while not self.is_running():
            time.sleep(0.1)

    @override
    def stop(self) -> str | None:
        while self.is_running():
            time.sleep(0.1)
        self.java_gateway.close()
        return None

    # --------------------- Simulator Status -------------------------------------------------------------------------

    @override
    def is_running(self) -> bool:
        try:
            # Try to do a simple operation to check if the connection is still alive
            self.java_gateway.jvm.java.lang.System.currentTimeMillis()
            return True
        except Exception:
            return False

    # --------------------- Simulator Interaction ---------------------------------------------------------------------

    @override
    def reset(self) -> Any:
        if self.is_running():
            self.stop()
        self.start()

        return self.env_connector.reset()

    @override
    def step(self, action_creator: Callable[[Any], Any]) -> Any:
        if not self.is_running():
            raise Exception("Simulator is not running")

        action = action_creator(self.java_gateway.jvm)
        return self.env_connector.step(action)
