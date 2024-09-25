package org.example.api.scheduler.gym;

import lombok.NonNull;
import org.example.api.dtos.TaskDto;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;
import org.example.api.scheduler.WorkflowScheduler;
import org.example.api.scheduler.gym.mappers.GymWorkflowMapper;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/// A workflow scheduler implementation that uses the Gym environment.
public class GymWorkflowScheduler<TObservation, TAction> implements WorkflowScheduler {
    private final GymWorkflowMapper<TObservation, TAction> algorithm;

    private final GymEnvironment<TObservation, TAction> environment;
    private final Map<VmId, VmDto> vms = new HashMap<>();
    private final Map<WorkflowTaskId, TaskDto> tasks = new HashMap<>();

    public GymWorkflowScheduler(@NonNull GymWorkflowMapper<TObservation, TAction> algorithm,
                                @NonNull GymSharedQueue<TObservation, TAction> queue) {
        this.algorithm = algorithm;
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
        var assignment = algorithm.schedule(environment, List.copyOf(vms.values()), List.copyOf(tasks.values()));
        assignment.ifPresent(vmAssignmentDto -> tasks.remove(new WorkflowTaskId(vmAssignmentDto.getWorkflowId(), vmAssignmentDto.getTaskId())));
        return assignment;
    }

    /// A private record to represent a VM ID.
    private record VmId(int vmId) {
    }

    /// A private record to represent a workflow task ID.
    private record WorkflowTaskId(int workflowId, int taskId) {
    }
}
