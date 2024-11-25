import json
import sys
import time
import socket
import subprocess

from typing import Any, Callable
from py4j.java_gateway import JavaGateway, GatewayParameters

from scheduler.dataset_generator.core.models import Dataset


# Taken from Selenium's utils.py
# https://github.com/SeleniumHQ/selenium/blob/35dd34afbdd96502066d0f7b6a2460a11e5fb73a/py/selenium/webdriver/common/utils.py#L31
def free_port() -> int:
    """Determines a free port using sockets."""
    free_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free_socket.bind(("127.0.0.1", 0))
    free_socket.listen(5)
    port: int = free_socket.getsockname()[1]
    free_socket.close()
    return port


class EmbeddedCloudSimSimulator:
    """
    A simulator that runs CloudSim using a Java process embedded in the Python process.
    This simulator starts and stops the Java process when the simulator is started and stopped.

    The simulator communicates with the Java process using the Py4J library, which allows calling Java methods
    from Python and vice versa. The simulator uses a Java gateway to connect to the Java process and send commands.
    """

    simulator_process: subprocess.Popen | None
    java_gateway: JavaGateway
    current_dataset: Dataset | None

    def __init__(
        self,
        simulator_jar_path: str,
        verbose: bool = False,
        remote_debug: bool = False,
    ):
        self.simulator_jar_path = simulator_jar_path
        self.simulator_process = None

        gateway_params = GatewayParameters(port=free_port())
        self.java_gateway = JavaGateway(gateway_parameters=gateway_params)
        self.env_connector = self.java_gateway.entry_point

        self.verbose = verbose
        self.remote_debug = remote_debug
        self.current_dataset = None

    # Simulator Start
    # ------------------------------------------------------------------------------------------------------------------

    def start(self, dataset: Dataset):
        self._verify_stopped()
        self._verify_port_free()

        # Start the simulator process
        port = self.java_gateway.gateway_parameters.port
        java_args = ["-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005"] if self.remote_debug else []

        self.simulator_process = subprocess.Popen(
            ["java", *java_args, "-jar", self.simulator_jar_path, "-p", str(port), "-a", "static:gym"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True,
        )
        self.current_dataset = dataset

        assert self.simulator_process.stdin is not None
        self.simulator_process.stdin.write(json.dumps(dataset.to_json()) + "\n")
        self.simulator_process.stdin.flush()
        if self.remote_debug:
            # The first line is the JVM listening message
            assert self.simulator_process.stdout is not None
            print(self.simulator_process.stdout.readline())

        # Wait for the simulator to start
        time.sleep(0.1)
        while not self.is_running():
            time.sleep(0.1)
        self._print_if_verbose(f"Simulator started with PID: {self.simulator_process.pid} on port {port}")

    # Simulator Stop
    # ------------------------------------------------------------------------------------------------------------------

    def stop(self) -> str | None:
        self._verify_running()

        # Terminate the simulator process
        assert self.simulator_process is not None
        self.java_gateway.close()
        self.simulator_process.terminate()
        self.simulator_process.wait()

        # Wait for the simulator to stop
        time.sleep(0.1)
        while self.is_running():
            time.sleep(0.1)
        self._print_if_verbose(f"Simulator stopped with PID: {self.simulator_process.pid}")

        assert self.simulator_process.stdout is not None
        assert self.simulator_process.stderr is not None
        output = self.simulator_process.stdout.read()
        self._print_if_verbose(self.simulator_process.stderr.read(), file=sys.stderr)
        self.simulator_process = None
        return output

    # Simulator Status
    # ------------------------------------------------------------------------------------------------------------------

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

    # Simulator Control
    # ------------------------------------------------------------------------------------------------------------------

    def reset(self, dataset: Dataset) -> Any:
        if self.is_running():
            self.stop()
        self.start(dataset)

        return self.env_connector.reset()

    def step(self, action_creator: Callable[[Any], Any]) -> Any:
        self._verify_running()

        action = action_creator(self.java_gateway.jvm)
        return self.env_connector.step(action)

    # Private Helpers
    # ------------------------------------------------------------------------------------------------------------------

    def _verify_running(self):
        if not self.is_running():
            raise Exception("Simulator is not running")

    def _verify_stopped(self):
        if self.is_running():
            raise Exception("Simulator is still running")

    def _verify_port_free(self):
        address = self.java_gateway.gateway_parameters.address
        port = self.java_gateway.gateway_parameters.port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            res = s.connect_ex((address, port))
        if res == 0:
            raise Exception(f"Port {port} is already in use")

    def _print_if_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
