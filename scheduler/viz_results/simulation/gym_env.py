import json

import gymnasium as gym
from typing import Any

from scheduler.dataset_generator.core.gen_dataset import generate_dataset
from scheduler.dataset_generator.core.models import Solution
from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.rl_model.core.types import TaskDto, VmDto
from scheduler.viz_results.simulation.cloudsim import EmbeddedCloudSimSimulator
from scheduler.viz_results.simulation.observation import SimEnvObservation, SimEnvAction


class CloudSimGymEnvironment(gym.Env):
    simulator: EmbeddedCloudSimSimulator
    last_obs: SimEnvObservation | None = None

    def __init__(self, simulator_jar_path: str, dataset_args: DatasetArgs, verbose=False, remote_debug=False):
        super().__init__()
        self.simulator = EmbeddedCloudSimSimulator(simulator_jar_path, verbose, remote_debug)
        self.dataset_args = dataset_args

    # Reset
    # ------------------------------------------------------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[SimEnvObservation, dict[str, Any]]:
        super().reset(seed=seed)

        dataset = generate_dataset(
            seed=seed if seed is not None else self.dataset_args.seed,
            host_count=self.dataset_args.host_count,
            vm_count=self.dataset_args.vm_count,
            max_memory_gb=self.dataset_args.max_memory_gb,
            min_cpu_speed_mips=self.dataset_args.min_cpu_speed,
            max_cpu_speed_mips=self.dataset_args.max_cpu_speed,
            workflow_count=self.dataset_args.workflow_count,
            dag_method=self.dataset_args.dag_method,
            gnp_min_n=self.dataset_args.gnp_min_n,
            gnp_max_n=self.dataset_args.gnp_max_n,
            task_length_dist=self.dataset_args.task_length_dist,
            min_task_length=self.dataset_args.min_task_length,
            max_task_length=self.dataset_args.max_task_length,
            task_arrival=self.dataset_args.task_arrival,
            arrival_rate=self.dataset_args.arrival_rate,
        )

        # Send the reset command to the simulator
        result = self.simulator.reset(dataset)
        obs = self.parse_obs(result)
        info: dict[str, Any] = {}

        return obs, info

    # Step
    # ------------------------------------------------------------------------------------------------------------------

    def step(self, action: SimEnvAction) -> tuple[SimEnvObservation, float, bool, bool, dict[str, Any]]:
        # Send the action to the simulator
        result = self.simulator.step(lambda jvm: self.create_action(jvm, action))
        obs = self.parse_obs(result.getObservation())
        reward = float(result.getReward())
        terminated = bool(result.isTerminated())
        truncated = bool(result.isTruncated())
        info: dict[str, Any] = self.parse_info(result.getInfo())

        if terminated or truncated:
            output = self.simulator.stop()
            info["stdout"] = output

        return obs, reward, terminated, truncated, info

    # Parse Observation
    # ------------------------------------------------------------------------------------------------------------------

    def parse_obs(self, obs: Any | None) -> SimEnvObservation:
        if obs is None:
            if self.last_obs is not None:
                return self.last_obs
            return SimEnvObservation(tasks=[], vms=[])

        return SimEnvObservation(
            tasks=[
                TaskDto(
                    id=int(task.getId()),
                    workflow_id=int(task.getWorkflowId()),
                    length=int(task.getLength()),
                    req_memory_mb=int(task.getReqMemoryMb()),
                    child_ids=[int(child) for child in task.getChildIds()],
                )
                for task in obs.getTasks()
            ],
            vms=[
                VmDto(
                    id=int(vm.getId()),
                    memory_mb=int(vm.getMemoryMb()),
                    cpu_speed_mips=float(vm.getCpuSpeedMips()),
                    host_power_idle_watt=float(vm.getHost().getPowerIdleWatt()),
                    host_power_peak_watt=float(vm.getHost().getPowerPeakWatt()),
                    host_cpu_speed_mips=float(vm.getHost().getCpuSpeedMips()),
                )
                for vm in obs.getVms()
            ],
        )

    # Parse Info
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def parse_info(info: Any | None) -> dict[str, Any]:
        raw_info = {} if info is None else {str(o.getKey()): str(o.getValue()) for o in info.entrySet()}
        parsed_info: dict[str, Any] = {}

        if raw_info.get("solution") is not None:
            solution_json = raw_info["solution"]
            solution_dict = json.loads(solution_json)
            parsed_info["solution"] = Solution.from_json(solution_dict)
        if raw_info.get("total_energy_consumption_j") is not None:
            power_cons_str = raw_info["total_energy_consumption_j"]
            parsed_info["total_energy_consumption_j"] = float(power_cons_str)

        return parsed_info

    # Parse Action
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def create_action(jvm: Any, action: SimEnvAction) -> Any:
        assignments = jvm.java.util.ArrayList()
        for act in action.vm_assignments:
            assignment = jvm.org.example.api.dtos.VmAssignmentDto(act.vm_id, act.workflow_id, act.task_id)
            assignments.add(assignment)
        return jvm.org.example.api.scheduler.gym.types.StaticAction(assignments)
