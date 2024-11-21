import random
import sys
import json
import time
import socket
import subprocess
import dataclasses
import hashlib

from typing import Any, Callable
from py4j.java_gateway import JavaGateway, GatewayParameters

from dataset_generator.core.models import Dataset
from gym_simulator.core.simulators.base import BaseSimulator
from dataset_generator.core.gen_dataset import generate_dataset
from dataset_generator.gen_dataset import Args as DatasetArgs


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


class EmbeddedSimulator(BaseSimulator):
    """
    A simulator that runs CloudSim using a Java process embedded in the Python process.
    This simulator starts and stops the Java process when the simulator is started and stopped.

    The simulator communicates with the Java process using the Py4J library, which allows calling Java methods
    from Python and vice versa. The simulator uses a Java gateway to connect to the Java process and send commands.
    """

    simulator_process: subprocess.Popen | None
    java_gateway: JavaGateway
    dataset_seed: int | None
    current_dataset: Dataset | None

    def __init__(
        self,
        simulator_jar_path: str,
        scheduler_preset: str,
        dataset_args: dict[str, Any],
        verbose: bool = False,
        remote_debug: bool = False,
    ):
        self.simulator_jar_path = simulator_jar_path
        self.scheduler_preset = scheduler_preset
        self.dataset_args = dataset_args
        self.simulator_process = None

        gateway_params = GatewayParameters(port=free_port())
        self.java_gateway = JavaGateway(gateway_parameters=gateway_params)
        self.env_connector = self.java_gateway.entry_point

        self.verbose = verbose
        self.remote_debug = remote_debug
        self.dataset_seed = None
        self.current_dataset = None

    # --------------------- Simulator Start ---------------------------------------------------------------------------

    def start(self):
        self._verify_stopped()
        self._verify_port_free()

        # Start the simulator process
        port = self.java_gateway.gateway_parameters.port
        java_args = ["-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005"] if self.remote_debug else []

        self.simulator_process = subprocess.Popen(
            ["java", *java_args, "-jar", self.simulator_jar_path, "-p", str(port), "-a", self.scheduler_preset],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True,
        )
        dataset_json = self._generate_dataset_json()
        assert self.simulator_process.stdin is not None
        self.simulator_process.stdin.write(dataset_json + "\n")
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

    # --------------------- Simulator Stop ----------------------------------------------------------------------------

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

    # --------------------- Simulator Reboot --------------------------------------------------------------------------

    def reboot(self):
        if self.is_running():
            self.stop()

        self.simulator_process = None
        gateway_params = GatewayParameters(port=free_port())
        self.java_gateway = JavaGateway(gateway_parameters=gateway_params)
        self.env_connector = self.java_gateway.entry_point
        self.start()

    # --------------------- Simulator Status -------------------------------------------------------------------------

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

    def reset(self, seed: int | None) -> Any:
        if self.is_running():
            self.stop()
        self.dataset_seed = seed
        self.start()

        return self.env_connector.reset()

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

    def _verify_port_free(self):
        address = self.java_gateway.gateway_parameters.address
        port = self.java_gateway.gateway_parameters.port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            res = s.connect_ex((address, port))
        if res == 0:
            raise Exception(f"Port {port} is already in use")

    def _generate_dataset_json(self) -> str:
        default_args = DatasetArgs()
        dataset = generate_dataset(
            seed=random.randint(1, 2**31) if self.dataset_seed is None else self.dataset_seed,
            host_count=self.dataset_args.get("host_count", default_args.host_count),
            vm_count=self.dataset_args.get("vm_count", default_args.vm_count),
            max_memory_gb=self.dataset_args.get("max_memory_gb", default_args.max_memory_gb),
            min_cpu_speed_mips=self.dataset_args.get("min_cpu_speed", default_args.min_cpu_speed),
            max_cpu_speed_mips=self.dataset_args.get("max_cpu_speed", default_args.max_cpu_speed),
            workflow_count=self.dataset_args.get("workflow_count", default_args.workflow_count),
            dag_method=self.dataset_args.get("dag_method", default_args.dag_method),
            gnp_min_n=self.dataset_args.get("gnp_min_n", default_args.gnp_min_n),
            gnp_max_n=self.dataset_args.get("gnp_max_n", default_args.gnp_max_n),
            task_length_dist=self.dataset_args.get("task_length_dist", default_args.task_length_dist),
            min_task_length=self.dataset_args.get("min_task_length", default_args.min_task_length),
            max_task_length=self.dataset_args.get("max_task_length", default_args.max_task_length),
            task_arrival=self.dataset_args.get("task_arrival", default_args.task_arrival),
            arrival_rate=self.dataset_args.get("arrival_rate", default_args.arrival_rate),
        )
        self.current_dataset = dataset
        json_str = json.dumps(dataclasses.asdict(dataset))
        hash_obj = hashlib.md5(json_str.encode())
        hash_str = hash_obj.hexdigest()
        self._print_if_verbose(f"Generated dataset with hash: {hash_str}")
        return json_str

    def _print_if_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
