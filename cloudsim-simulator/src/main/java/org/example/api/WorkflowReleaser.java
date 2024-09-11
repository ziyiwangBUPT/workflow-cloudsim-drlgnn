package org.example.api;

import lombok.NonNull;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;

import java.util.List;

/// The interface for releasing workflows depending on the current state of the system.
public interface WorkflowReleaser {
    /// Submit a list of VMs to the system.
    /// This is most likely called when the system is initialized.
    void submitVms(@NonNull List<VmDto> newVms);

    /// Submit a list of workflows to the system.
    /// This is most likely called when the broker receives new workflows.
    void submitWorkflows(@NonNull List<WorkflowDto> newWorkflows);

    /// Check if the system should release workflows.
    /// This is called periodically.
    boolean shouldRelease();
}
