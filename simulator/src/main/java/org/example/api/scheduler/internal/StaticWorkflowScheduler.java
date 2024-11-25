package org.example.api.scheduler.internal;

import lombok.NonNull;
import org.example.api.dtos.TaskDto;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;
import org.example.api.scheduler.WorkflowScheduler;
import org.example.api.scheduler.internal.algorithms.StaticSchedulingAlgorithm;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/// Static workflow scheduler is a scheduler that schedules workflows statically.
/// Each workflow is assigned to a VM at the beginning, and it does not change.
public class StaticWorkflowScheduler implements WorkflowScheduler {
    private final StaticSchedulingAlgorithm algorithm;

    private final List<VmDto> vms = new ArrayList<>();
    private final List<WorkflowDto> workflows = new ArrayList<>();
    private List<VmAssignmentDto> schedulingResult = null;

    public StaticWorkflowScheduler(@NonNull StaticSchedulingAlgorithm algorithm) {
        this.algorithm = algorithm;
    }

    @Override
    public void notifyNewVm(@NonNull VmDto newVm) {
        if (schedulingResult != null) {
            throw new IllegalStateException("Cannot add new VMs after scheduling.");
        }
        vms.add(newVm);
    }

    @Override
    public void notifyNewWorkflow(@NonNull WorkflowDto newWorkflow) {
        if (schedulingResult != null) {
            throw new IllegalStateException("Cannot add new workflows after scheduling.");
        }
        workflows.add(newWorkflow);
    }

    @Override
    public Optional<VmAssignmentDto> schedule() {
        if (schedulingResult == null) {
            if (workflows.isEmpty() || vms.isEmpty()) {
                return Optional.empty();
            }
            var tasks = new ArrayList<TaskDto>();
            workflows.forEach(workflow -> tasks.addAll(workflow.getTasks()));
            schedulingResult = algorithm.schedule(tasks, vms);
        }

        if (schedulingResult.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(schedulingResult.removeFirst());
    }
}
