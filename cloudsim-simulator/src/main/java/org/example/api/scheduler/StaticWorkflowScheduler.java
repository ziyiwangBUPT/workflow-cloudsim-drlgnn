package org.example.api.scheduler;

import lombok.NonNull;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/// The abstract class for the static workflow scheduler.
/// Static workflow scheduler is a scheduler that schedules workflows statically.
/// Each workflow is assigned to a VM at the beginning, and it does not change.
public abstract class StaticWorkflowScheduler implements WorkflowScheduler {
    private final List<VmDto> vms = new ArrayList<>();
    private final List<WorkflowDto> workflows = new ArrayList<>();
    private List<VmAssignmentDto> schedulingResult = null;

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
            schedulingResult = schedule(workflows, vms);
        }

        if (schedulingResult.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(schedulingResult.removeFirst());
    }

    /// Schedules the workflows to the VMs.
    /// This is the method that the subclasses should implement.
    /// This is called only once.
    protected abstract List<VmAssignmentDto> schedule(List<WorkflowDto> workflows, List<VmDto> vms);
}
