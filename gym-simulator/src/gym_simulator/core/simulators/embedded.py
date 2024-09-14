import time
import subprocess

from typing import Any, Callable, override
from gym_simulator.core.simulators.base import BaseSimulator
from py4j.java_gateway import JavaGateway


class EmbeddedSimulator(BaseSimulator):
    """
    A simulator that runs CloudSim using a Java process embedded in the Python process.
    This simulator starts and stops the Java process when the simulator is started and stopped.

    The simulator communicates with the Java process using the Py4J library, which allows calling Java methods
    from Python and vice versa. The simulator uses a Java gateway to connect to the Java process and send commands.
    """

    simulator_process: subprocess.Popen | None
    java_gateway: JavaGateway

    def __init__(self, simulator_jar_path: str, dataset_path: str):
        self.simulator_jar_path = simulator_jar_path
        self.dataset_path = dataset_path
        self.simulator_process = None
        self.java_gateway = JavaGateway()
        self.env_connector = self.java_gateway.entry_point

    # --------------------- Simulator Start ---------------------------------------------------------------------------

    @override
    def start(self):
        print("Starting the simulator...")
        self._verify_stopped()

        # Start the simulator process
        self.simulator_process = subprocess.Popen(
            ["java", "-jar", self.simulator_jar_path, "-f", self.dataset_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        # Wait for the simulator to start
        while not self.is_running():
            time.sleep(0.1)
        print(f"Simulator started with PID: {self.simulator_process.pid}")

    # --------------------- Simulator Stop ----------------------------------------------------------------------------

    @override
    def stop(self):
        print("Stopping the simulator...")
        self._verify_running()

        # Terminate the simulator process
        assert self.simulator_process is not None
        self.java_gateway.close()
        self.simulator_process.terminate()
        self.simulator_process.wait()
        self.simulator_process = None

        # Wait for the simulator to stop
        while self.is_running():
            time.sleep(0.1)
        print("Simulator stopped")

    # --------------------- Simulator Status -------------------------------------------------------------------------

    @override
    def is_running(self) -> bool:
        if self.simulator_process is None:
            return False
        if self.simulator_process.poll() is not None:
            return False

        try:
            # Try to do a simple operation to check if the connection is still alive
            self.java_gateway.jvm.java.lang.System.currentTimeMillis()
            return True
        except Exception:
            return False

    # --------------------- Simulator Control -------------------------------------------------------------------------

    @override
    def reset(self) -> Any:
        if self.is_running():
            self.stop()
        self.start()

        return self.env_connector.reset()

    @override
    def step(self, action_creator: Callable[[Any], Any]) -> Any:
        self._verify_running()

        action = action_creator(self.java_gateway.jvm)
        return self.env_connector.step(action)

    # --------------------- Private Helpers --------------------------------------------------------------------------

    def _verify_running(self):
        if not self.is_running():
            raise Exception("Simulator is not running")

    def _verify_stopped(self):
        if self.is_running():
            raise Exception("Simulator is still running")
