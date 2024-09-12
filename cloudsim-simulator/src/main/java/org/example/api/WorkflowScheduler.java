package org.example.api;

import lombok.NonNull;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;

import java.util.Optional;

/// The interface for the workflow scheduler.
public interface WorkflowScheduler {
    /// Submit a new VM to the system.
    /// This is called from the coordinator when it discovers a new VM.
    void notifyNewVm(@NonNull VmDto newVm);

    /// Submit a new workflow to the system.
    /// This is called from the coordinator when a workflow gets released.
    void notifyNewWorkflow(@NonNull WorkflowDto newWorkflow);

    /// This is called periodically from the coordinator to schedule a workflow.
    /// If there is no scheduling decision to be made, this will return an empty optional.
    Optional<VmAssignmentDto> schedule();
}
