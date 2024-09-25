package org.example.api.scheduler.impl;

import lombok.NonNull;
import org.example.api.dtos.TaskDto;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;
import org.example.api.scheduler.WorkflowScheduler;
import org.example.api.scheduler.gym.GymEnvironment;
import org.example.api.scheduler.gym.GymSharedQueue;
import org.example.api.scheduler.gym.types.Action;
import org.example.api.scheduler.gym.types.AgentResult;
import org.example.api.scheduler.gym.types.Observation;
import org.example.utils.GsonHelper;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/// A workflow scheduler implementation that uses the Gym environment.
public class GymWorkflowScheduler implements WorkflowScheduler {
    private final GymEnvironment<Observation, Action> environment;
    private final Map<VmId, VmDto> vms = new HashMap<>();
    private final Map<WorkflowTaskId, TaskDto> tasks = new HashMap<>();

    public GymWorkflowScheduler(GymSharedQueue<Observation, Action> queue) {
        this.environment = new GymEnvironment<>(queue);
    }

    @Override
    public void notifyNewVm(@NonNull VmDto newVm) {
        vms.put(new VmId(newVm.getId()), newVm);
    }

    @Override
    public void notifyNewWorkflow(@NonNull WorkflowDto newWorkflow) {
        for (var task : newWorkflow.getTasks()) {
            tasks.put(new WorkflowTaskId(newWorkflow.getId(), task.getId()), task);
        }
    }

    @Override
    public Optional<VmAssignmentDto> schedule() {
        // Observation
        var gson = GsonHelper.getGson();
        var vmJson = gson.toJson(vms.values());
        var taskJson = gson.toJson(tasks.values());
        var observation = new Observation(vmJson, taskJson);

        // Reward
        var reward = 0.0;

        // Step
        var action = environment.step(AgentResult.reward(observation, reward));
        if (action == null || action.isNoOp()) {
            return Optional.empty();
        }
        return Optional.of(new VmAssignmentDto(action.getVmId(), action.getWorkflowId(), action.getTaskId()));
    }

    /// A private record to represent a VM ID.
    private record VmId(int vmId) {
    }

    /// A private record to represent a workflow task ID.
    private record WorkflowTaskId(int workflowId, int taskId) {
    }
}
