package org.example.api.scheduler;

import lombok.NonNull;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;

/// The interface for releasing workflows depending on the current state of the system.
public interface WorkflowReleaser {
    /// Submit a new VM to the system.
    /// This is called from the coordinator when it discovers a new VM.
    void notifyNewVm(@NonNull VmDto newVm);

    /// Submit a new workflow to the system.
    /// This is called from the workflow buffer when a new workflow arrives.
    void notifyNewWorkflow(@NonNull WorkflowDto newWorkflow);

    /// Check if the system should release workflows.
    /// This is called periodically from the workflow buffer to decide if the system should release workflows.
    boolean shouldRelease();
}
