import json
import time
import socket
import subprocess
import dataclasses

from typing import Any, Callable, override
from py4j.java_gateway import JavaGateway, DEFAULT_PORT, DEFAULT_ADDRESS, GatewayParameters

from gym_simulator.core.simulators.base import BaseSimulator
from dataset_generator.core.gen_dataset import generate_dataset
from dataset_generator.gen_dataset import Args as DatasetArgs


class EmbeddedSimulator(BaseSimulator):
    """
    A simulator that runs CloudSim using a Java process embedded in the Python process.
    This simulator starts and stops the Java process when the simulator is started and stopped.

    The simulator communicates with the Java process using the Py4J library, which allows calling Java methods
    from Python and vice versa. The simulator uses a Java gateway to connect to the Java process and send commands.
    """

    simulator_process: subprocess.Popen | None
    java_gateway: JavaGateway

    def __init__(self, jvm_port: int, simulator_jar_path: str, dataset_args: dict[str, Any]):
        self.simulator_jar_path = simulator_jar_path
        self.dataset_args = dataset_args
        self.simulator_process = None

        gateway_params = GatewayParameters(port=jvm_port)
        self.java_gateway = JavaGateway(gateway_parameters=gateway_params)
        self.env_connector = self.java_gateway.entry_point

    # --------------------- Simulator Start ---------------------------------------------------------------------------

    @override
    def start(self):
        self._verify_stopped()
        self._verify_port_free()

        # Start the simulator process
        port = self.java_gateway.gateway_parameters.port
        print(f"Starting the simulator on port {port}...")
        self.simulator_process = subprocess.Popen(
            ["java", "-jar", self.simulator_jar_path, "-p", str(port), "-a", "static:gym"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True,
        )
        dataset_json = self._generate_dataset_json()
        assert self.simulator_process.stdin is not None
        self.simulator_process.stdin.write(dataset_json + "\n")

        # Wait for the simulator to start
        time.sleep(0.1)
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
        time.sleep(0.1)
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
            host_count=self.dataset_args.get("host_count", default_args.host_count),
            vm_count=self.dataset_args.get("vm_count", default_args.vm_count),
            max_cores=self.dataset_args.get("max_cores", default_args.max_cores),
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
        return json.dumps(dataclasses.asdict(dataset))
