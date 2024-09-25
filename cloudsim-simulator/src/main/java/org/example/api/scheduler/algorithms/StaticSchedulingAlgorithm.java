package org.example.api.scheduler.algorithms;

import lombok.NonNull;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;

import java.util.List;

/// A static scheduling algorithm.
/// Static scheduling algorithm is an algorithm that schedules workflows with a fixed schedule.
/// Arrival time of workflows and VMs are not considered.
public interface StaticSchedulingAlgorithm {
    /// Schedules the workflows to the VMs.
    List<VmAssignmentDto> schedule(@NonNull List<WorkflowDto> workflows, @NonNull List<VmDto> vms);
}
