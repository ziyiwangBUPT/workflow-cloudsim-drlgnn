import abc
import sys
import time
import subprocess


class SimulatorRunner(abc.ABC):
    def run(self):
        pass

    def stop(self):
        pass

    @abc.abstractmethod
    def is_running(self) -> bool:
        raise NotImplementedError


# --------------------- NoOpSimulatorRunner --------------------------------


class NoOpSimulatorRunner(SimulatorRunner):
    def is_running(self) -> bool:
        return True


# --------------------- CloudSimSimulatorRunner -----------------------------


class CloudSimSimulatorRunner(SimulatorRunner):
    simulator_process: subprocess.Popen | None = None

    def __init__(self, simulator: str, dataset: str):
        self.simulator = simulator
        self.dataset = dataset

    def run(self):
        self.simulator_process = subprocess.Popen(
            ["java", "-jar", self.simulator, "-f", self.dataset],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        time.sleep(1)
        if not self.is_running():
            raise Exception("Simulator failed to start")
        print(f"Simulator started with PID: {self.simulator_process.pid}")

    def stop(self):
        if self.is_running() and self.simulator_process is not None:
            self.simulator_process.terminate()
            self.simulator_process.wait()
            self.simulator_process = None

            print("Simulator stopped")

    def is_running(self) -> bool:
        return self.simulator_process is not None and self.simulator_process.poll() is None

    def get_output(self):
        if self.simulator_process is not None and self.simulator_process.stdout is not None:
            with open(self.simulator_process.stdout.fileno(), "r") as f:
                return f.read()
