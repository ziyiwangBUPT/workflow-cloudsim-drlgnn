package org.example.api;

import lombok.NonNull;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;

import java.util.List;

/// The interface for the workflow scheduler.
public interface WorkflowScheduler {
    /// Schedules the workflows on the VMs.
    /// Performs static scheduling.
    List<VmAssignmentDto> schedule(@NonNull List<WorkflowDto> workflows, @NonNull List<VmDto> vms);
}
