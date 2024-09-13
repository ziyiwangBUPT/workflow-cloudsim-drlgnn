import sys
import time
import subprocess


class SimulatorRunner:
    simulator_process: subprocess.Popen | None = None

    def __init__(self, simulator: str, dataset: str):
        self.simulator = simulator
        self.dataset = dataset

    def run(self):
        """Run the simulator parallely"""
        self.simulator_process = subprocess.Popen(
            ["java", "-jar", self.simulator, "-f", self.dataset],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        time.sleep(5)
        if self.simulator_process.poll() is not None:
            print("Simulator failed to start")
            sys.exit(1)

        print(f"Simulator started with PID: {self.simulator_process.pid}")

    def stop(self):
        """Stop the simulator"""
        if self.simulator_process is not None:
            self.simulator_process.terminate()
            self.simulator_process.wait()
            self.simulator_process = None

            print("Simulator stopped")

    def is_running(self) -> bool:
        """Check if the simulator is running"""
        return self.simulator_process is not None and self.simulator_process.poll() is None
