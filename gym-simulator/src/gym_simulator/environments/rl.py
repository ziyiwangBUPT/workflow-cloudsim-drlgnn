import copy

import dataclasses
from typing import Any
from gymnasium import spaces
import numpy as np

from dataset_generator.core.models import Dataset
from gym_simulator.algorithms.heft_one import HeftOneScheduler
from gym_simulator.algorithms.round_robin import RoundRobinScheduler
from gym_simulator.core.simulators.embedded import EmbeddedSimulator
from gym_simulator.core.types import TaskDto, VmAssignmentDto, VmDto
from gym_simulator.environments.basic import BasicCloudSimEnvironment
from gym_simulator.environments.states.rl import RlEnvState
from gym_simulator.renderers.rl import RlEnvironmentRenderer
from gym_simulator.utils import makespan_calculator
from gym_simulator.utils.task_mapper import TaskMapper


class RlCloudSimEnvironment(BasicCloudSimEnvironment):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    state: RlEnvState | None = None

    # ----------------------- Initialization --------------------------------------------------------------------------

    def __init__(self, env_config: dict[str, Any]):
        assert (
            env_config["simulator_mode"] in ["embedded", "internal"],
            "Only embedded or internal simulator modes are supported",
        )

        # Override args
        simulator_kwargs = env_config.get("simulator_kwargs", {})
        simulator_kwargs["dataset_args"] = simulator_kwargs.get("dataset_args", {})
        assert "task_arrival" not in simulator_kwargs["dataset_args"], "task_arrival is set by the environment"
        simulator_kwargs["dataset_args"]["task_arrival"] = "static"

        super().__init__(env_config)
        self.parent_observation_space = copy.deepcopy(self.observation_space)
        self.parent_action_space = copy.deepcopy(self.action_space)
        self.renderer = RlEnvironmentRenderer(render_fps=self.metadata["render_fps"], width=1200, height=800)

        max_vm_count = self.vm_count
        # Reserve 0 for dummy start, N+1 for dummy end, Nmax=W*T
        max_task_count = self.workflow_count * self.task_limit + 2

        self.observation_space = spaces.Dict(
            {
                # Scheduling state: 0 - not scheduled, 1 - scheduled, shape: (num_tasks,)
                "task_state_scheduled": spaces.MultiBinary(max_task_count),
                # Ready state: 0 - not ready, 1 - ready, shape: (num_tasks,)
                "task_state_ready": spaces.MultiBinary(max_task_count),
                # Est min completion time for unscheduled tasks or completion time for scheduled tasks, shape: (num_tasks,)
                "task_completion_time": spaces.Box(low=0, high=np.inf, shape=(max_task_count,)),
                # Machine completion time for each VM, shape: (num_vms,)
                "vm_completion_time": spaces.Box(low=0, high=np.inf, shape=(max_vm_count,)),
                # Task-VM compatibility: 0 - not compatible, 1 - compatible, shape: (num_tasks, num_vms)
                "task_vm_compatibility": spaces.MultiBinary((max_task_count, max_vm_count)),
                # Task-VM time cost: (average of compatible tasks if not compatible), shape: (num_tasks, num_vms)
                "task_vm_time_cost": spaces.Box(low=0, high=np.inf, shape=(max_task_count, max_vm_count)),
                # Task-VM power cost: (average of compatible tasks if not compatible), shape: (num_tasks, num_vms)
                "task_vm_power_cost": spaces.Box(low=0, high=np.inf, shape=(max_task_count, max_vm_count)),
                # Task row->col fixed dependencies: 0 - no dependency, 1 - dependency, shape: (num_tasks, num_tasks)
                "task_graph_edges": spaces.MultiBinary((max_task_count, max_task_count)),
            }
        )
        self.action_space = spaces.Dict(
            {
                # Task ID to schedule, shape: ()
                "task_id": spaces.Discrete(max_task_count),
                # VM ID to schedule the task on, shape: ()
                "vm_id": spaces.Discrete(max_vm_count),
            }
        )

    # ----------------------- Reset method ----------------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self.state is not None:
            self.simulator.reboot()
            self.state = None

        dict_observation, info = super().reset(seed=seed, options=options)
        tasks = [TaskDto(**task) for task in dict_observation["tasks"]]
        vms = [VmDto(**vm) for vm in dict_observation["vms"]]

        task_mapper = TaskMapper(tasks)
        mapped_tasks = task_mapper.map_tasks()

        # Sanity check - we should be able to use index and id interchangeably
        for i, task in enumerate(mapped_tasks):
            assert task.id == i, f"Task ID mismatch: {task.id} != {i}"
        for i, vm in enumerate(vms):
            assert vm.id == i, f"VM ID mismatch: {vm.id} != {i}"

        # Initialize the state
        task_state_scheduled = self._init_task_state_scheduled(mapped_tasks)
        task_state_ready = self._init_task_state_ready(mapped_tasks)
        compatibility_and_costs = self._init_compatibility_and_costs(mapped_tasks, vms)
        task_vm_compatibility, task_vm_time_cost, task_vm_power_cost = compatibility_and_costs
        task_completion_time = self._init_task_completion_time(mapped_tasks, task_vm_time_cost)
        vm_completion_time = self._init_vm_completion_time(vms)
        task_graph_edges = self._init_task_graph_edges(mapped_tasks)
        assignments = self._init_assignments(mapped_tasks)
        self.state = RlEnvState(
            task_mapper=task_mapper,
            task_state_scheduled=task_state_scheduled,
            task_state_ready=task_state_ready,
            task_completion_time=task_completion_time,
            vm_completion_time=vm_completion_time,
            task_vm_compatibility=task_vm_compatibility,
            task_vm_time_cost=task_vm_time_cost,
            task_vm_power_cost=task_vm_power_cost,
            task_graph_edges=task_graph_edges,
            assignments=assignments,
            tasks=mapped_tasks,
        )

        # Update the renderer
        if self.render_mode == "human":
            self.render()

        return self.state.to_observation(), info

    # ----------------------- Step method ------------------------------------------------------------------------------

    def step(self, action: dict[str, int]) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.state is not None, "State must be initialized"

        task_id = action["task_id"]
        vm_id = action["vm_id"]

        # Validate action
        penalty = 10000 + 1000 * sum(self.state.task_state_scheduled == 0)
        if len(self.state.task_state_scheduled) <= task_id:
            return {}, -penalty, True, False, {"error": f"Task {task_id} is is a banned task"}
        if self.state.task_state_scheduled[task_id] == 1:
            return {}, -penalty, True, False, {"error": f"Task {task_id} is already scheduled"}
        if self.state.task_state_ready[task_id] == 0:
            return {}, -penalty, True, False, {"error": f"Task {task_id} is not ready"}
        if self.state.task_vm_compatibility[task_id, vm_id] == 0:
            return {}, -penalty, True, False, {"error": f"Task {task_id} is not compatible with VM {vm_id}"}

        mapped_tasks = self.state.tasks

        new_task_state_scheduled = self._update_task_state_scheduled(task_id)
        new_task_state_ready = self._update_task_state_ready(task_id, new_task_state_scheduled)
        new_vm_completion_time = self._update_vm_completion_time(task_id, vm_id)
        new_task_completion_time = self._update_task_completion_time(
            task_id, vm_id, mapped_tasks, new_vm_completion_time
        )
        new_task_graph_edges = self._update_task_graph_edges(task_id, vm_id)
        new_assignments = self._update_assignments(task_id, vm_id)

        # Update task assigned VM id
        new_assignments = self.state.assignments.copy()
        new_assignments[task_id] = vm_id

        if new_task_state_ready[-1] == 1:
            # Dummy end active, all tasks have been scheduled
            # Let's assign it to 0 vm (it doesnt matter since length is 0)
            new_task_state_scheduled[-1] = 1
            new_task_state_ready[-1] = 0
            new_assignments[-1] = 0

        old_makespan = self.state.task_completion_time[-1]
        new_makespan = new_task_completion_time[-1]
        self.state = RlEnvState(
            task_mapper=self.state.task_mapper,
            task_state_scheduled=new_task_state_scheduled,
            task_state_ready=new_task_state_ready,
            task_completion_time=new_task_completion_time,
            vm_completion_time=new_vm_completion_time,
            task_vm_compatibility=self.state.task_vm_compatibility,
            task_vm_time_cost=self.state.task_vm_time_cost,
            task_vm_power_cost=self.state.task_vm_power_cost,
            task_graph_edges=new_task_graph_edges,
            assignments=new_assignments,
            tasks=self.state.tasks,
        )

        # Update the renderer
        if self.render_mode == "human":
            self.render()

        if self.state.task_state_scheduled[-1] == 1:
            # Last task scheduled, lets submit the result
            combined_action: list[tuple[float, VmAssignmentDto]] = []
            for task_id, vm_id in enumerate(self.state.assignments):
                if task_id == 0 or task_id == len(self.state.assignments) - 1:
                    continue
                u_workflow_id, u_vm_id = self.state.task_mapper.unmap_id(task_id)
                comp_time = self.state.task_completion_time[task_id]
                combined_action.append((comp_time, VmAssignmentDto(int(vm_id), u_workflow_id, u_vm_id)))
            combined_action.sort(key=lambda x: x[0])
            dict_action = [dataclasses.asdict(a[1]) for a in combined_action]
            obs, _, terminated, truncated, info = super().step(dict_action)
            info["vm_assignments"] = [a[1] for a in combined_action]

            baseline_makespan = self._calculate_baseline_makespan()
            makespan = self.state.task_completion_time[-1]
            reward = -makespan / baseline_makespan

            return obs, reward, terminated, truncated, info

        immediate_reward = (old_makespan - new_makespan) * 1e-6
        return self.state.to_observation(), immediate_reward, False, False, {}

    # ----------------------- Rendering -------------------------------------------------------------------------------

    def render(self):
        if self.state is None:
            return None

        if self.render_mode == "human":
            assert self.renderer is not None, "Human rendering requires a renderer"
            self.renderer.update(self.state)
        elif self.render_mode == "rgb_array":
            assert self.renderer is not None, "RGB array rendering requires a renderer"
            return self.renderer.draw(self.state)
        return None

    # ----------------------- Initializing State ----------------------------------------------------------------------

    def _init_task_state_scheduled(self, mapped_tasks: list[TaskDto]) -> np.ndarray:
        # Initially only dummy start task is scheduled.
        task_state_scheduled = np.zeros(len(mapped_tasks), dtype=np.int64)
        task_state_scheduled[0] = 1
        return task_state_scheduled

    def _init_task_state_ready(self, mapped_tasks: list[TaskDto]) -> np.ndarray:
        # Initially only the children of dummy start task are ready.
        task_state_ready = np.zeros(len(mapped_tasks), dtype=np.int64)
        for child_id in mapped_tasks[0].child_ids:
            task_state_ready[child_id] = 1
        return task_state_ready

    def _init_compatibility_and_costs(
        self, mapped_tasks: list[TaskDto], vms: list[VmDto]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Updating compatibility, time and power cost for each task-vm pair
        task_vm_compatibility = np.zeros((len(mapped_tasks), len(vms)), dtype=np.int64)
        task_vm_time_cost = np.zeros((len(mapped_tasks), len(vms)), dtype=np.float64)
        task_vm_power_cost = np.zeros((len(mapped_tasks), len(vms)), dtype=np.float64)
        for task_id, task in enumerate(mapped_tasks):
            for vm_id, vm in enumerate(vms):
                if self._is_vm_suitable(vm, task):
                    task_vm_compatibility[task_id, vm_id] = 1
                    task_vm_time_cost[task_id, vm_id] = task.length / vm.cpu_speed_mips
                    host_capacity_frac = vm.cpu_speed_mips / vm.host_cpu_speed_mips
                    power_per_sec = host_capacity_frac * (vm.host_power_peak_watt - vm.host_power_idle_watt)
                    task_vm_power_cost[task_id, vm_id] = task_vm_time_cost[task_id, vm_id] * power_per_sec
        # Fill all 0 value cells with average time and power cost of 1 value cells in each row
        for task_id in range(len(mapped_tasks)):
            avg_time_cost = np.mean(task_vm_time_cost[task_id, task_vm_compatibility[task_id] == 1])
            avg_power_cost = np.mean(task_vm_power_cost[task_id, task_vm_compatibility[task_id] == 1])
            for vm_id in range(len(vms)):
                if task_vm_compatibility[task_id, vm_id] == 0:
                    task_vm_time_cost[task_id, vm_id] = avg_time_cost
                    task_vm_power_cost[task_id, vm_id] = avg_power_cost

        return (task_vm_compatibility, task_vm_time_cost, task_vm_power_cost)

    def _init_task_completion_time(self, mapped_tasks: list[TaskDto], time_costs: np.ndarray) -> np.ndarray:
        # Updating completion time for each task, this is initially est min completion time
        # As an equation, max(est comp time of parents) + (task length / max vm speed suitable for task)
        task_completion_time = np.zeros(len(mapped_tasks), dtype=np.float64)
        for task_id in range(len(mapped_tasks)):
            parents = [parent for parent in mapped_tasks if task_id in parent.child_ids]
            est_max_comp_time_of_parents = max((task_completion_time[parent.id] for parent in parents), default=0)
            est_min_exec_time = np.min(time_costs[task_id])
            task_completion_time[task_id] = est_max_comp_time_of_parents + est_min_exec_time

        return task_completion_time

    def _init_vm_completion_time(self, vms: list[VmDto]) -> np.ndarray:
        # Completion time for VMs is 0 initially (only dummy start task is scheduled - has 0 length)
        return np.zeros(len(vms), dtype=np.float64)

    def _init_task_graph_edges(self, mapped_tasks: list[TaskDto]) -> np.ndarray:
        # Updating task graph edges, this initially only contains parent->child dependencies
        # So if A has B as child, then (A, B) = 1
        task_graph_edges = np.zeros((len(mapped_tasks), len(mapped_tasks)), dtype=np.int64)
        for task_id, task in enumerate(mapped_tasks):
            for child_id in task.child_ids:
                task_graph_edges[task_id, child_id] = 1
        return task_graph_edges

    def _init_assignments(self, mapped_tasks: list[TaskDto]) -> np.ndarray:
        # Initially only dummy start task is assigned to VM 0
        assignments = np.zeros(len(mapped_tasks), dtype=np.int64)
        assignments[0] = 0
        return assignments

    # ----------------------- Updating State --------------------------------------------------------------------------

    def _update_task_state_scheduled(self, scheduled_task_id: int) -> np.ndarray:
        # Update task state scheduled
        assert self.state is not None, "State must be initialized"
        new_task_state_scheduled = self.state.task_state_scheduled.copy()
        new_task_state_scheduled[scheduled_task_id] = 1
        return new_task_state_scheduled

    def _update_task_state_ready(self, scheduled_task_id: int, task_state_scheduled: np.ndarray) -> np.ndarray:
        # Update task state ready
        assert self.state is not None, "State must be initialized"
        new_task_state_ready = self.state.task_state_ready.copy()
        new_task_state_ready[scheduled_task_id] = 0
        child_ids = np.where(self.state.task_graph_edges[scheduled_task_id] == 1)[0]
        for child_id in child_ids:
            parent_ids = np.where(self.state.task_graph_edges[:, child_id] == 1)[0]
            if all(task_state_scheduled[parent_ids] == 1):
                new_task_state_ready[child_id] = 1
        return new_task_state_ready

    def _update_vm_completion_time(self, scheduled_task_id: int, scheduled_vm_id: int) -> np.ndarray:
        # Task will complete either after parent or after VM completion time
        assert self.state is not None, "State must be initialized"
        parent_ids = np.where(self.state.task_graph_edges[:, scheduled_task_id] == 1)[0]
        max_comp_time_of_parents = max(self.state.task_completion_time[parent_ids], default=0)
        exec_time = self.state.task_vm_time_cost[scheduled_task_id, scheduled_vm_id]
        max_comp_time = max(max_comp_time_of_parents, self.state.vm_completion_time[scheduled_vm_id]) + exec_time

        # And update VM completion time
        new_vm_completion_time = self.state.vm_completion_time.copy()
        new_vm_completion_time[scheduled_vm_id] = max_comp_time
        return new_vm_completion_time

    def _update_task_completion_time(
        self, scheduled_task_id: int, scheduled_vm_id: int, mapped_tasks: list[TaskDto], vm_completion_time: np.ndarray
    ) -> np.ndarray:
        # Update task completion time (task will complete either after parent or after VM completion time)
        assert self.state is not None, "State must be initialized"
        new_task_completion_time = self.state.task_completion_time.copy()
        for task_id in range(len(mapped_tasks)):
            new_task_completion_time[task_id] = float("inf")
            if task_id == scheduled_task_id:
                # Task that was just scheduled
                new_task_completion_time[task_id] = vm_completion_time[scheduled_vm_id]
                continue
            if self.state.task_state_scheduled[task_id] == 1:
                # Already scheduled task
                new_task_completion_time[task_id] = self.state.task_completion_time[task_id]
                continue
            parent_ids = np.where(self.state.task_graph_edges[:, task_id] == 1)[0]
            est_max_comp_time_of_parents = max(new_task_completion_time[parent_ids], default=0)
            compatible_vm_ids = np.where(self.state.task_vm_compatibility[task_id] == 1)[0]
            for vm_id in compatible_vm_ids:
                vm_comp_time = vm_completion_time[vm_id]
                est_exec_time = self.state.task_vm_time_cost[task_id, vm_id]
                updated_comp_time = max(est_max_comp_time_of_parents, vm_comp_time) + est_exec_time
                new_task_completion_time[task_id] = min(updated_comp_time, new_task_completion_time[task_id])
        return new_task_completion_time

    def _update_task_graph_edges(self, scheduled_task_id: int, scheduled_vm_id: int) -> np.ndarray:
        # Update task graph edges
        assert self.state is not None, "State must be initialized"
        scheduled_task_ids = set(np.where(self.state.task_state_scheduled == 1)[0])
        task_ids_assigned_to_vm = set(np.where(self.state.assignments == scheduled_vm_id)[0])
        task_ids_scheduled_in_vm = scheduled_task_ids.intersection(task_ids_assigned_to_vm)
        task_ids_scheduled_in_vm.add(0)  # Dummy start task
        task_completion_time = self.state.task_completion_time
        task_with_last_comp_time = max(task_ids_scheduled_in_vm, key=lambda task_id: task_completion_time[task_id])
        new_task_graph_edges = self.state.task_graph_edges.copy()
        new_task_graph_edges[task_with_last_comp_time, scheduled_task_id] = 1
        return new_task_graph_edges

    def _update_assignments(self, scheduled_task_id: int, scheduled_vm_id: int) -> np.ndarray:
        assert self.state is not None, "State must be initialized"
        new_assignments = self.state.assignments.copy()
        new_assignments[scheduled_task_id] = scheduled_vm_id
        return new_assignments

    # ----------------------- Private methods -------------------------------------------------------------------------

    def _is_vm_suitable(self, vm: VmDto, task: TaskDto) -> bool:
        """Check if the VM is suitable for the task."""
        return vm.memory_mb >= task.req_memory_mb

    def _calculate_baseline_makespan(self):
        current_dataset = getattr(self.simulator, "current_dataset", None)
        assert isinstance(current_dataset, Dataset)

        tasks = [
            TaskDto(**dataclasses.asdict(task)) for workflow in current_dataset.workflows for task in workflow.tasks
        ]
        vms = [
            VmDto(
                id=vm.id,
                memory_mb=vm.memory_mb,
                cpu_speed_mips=vm.cpu_speed_mips,
                host_cpu_speed_mips=-1,
                host_power_peak_watt=-1,
                host_power_idle_watt=-1,
            )
            for vm in current_dataset.vms
        ]
        assignments = HeftOneScheduler().schedule(tasks, vms)
        return makespan_calculator.makespan_calculator(tasks, vms, assignments)
