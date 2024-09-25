package org.example.api.scheduler.gym.mappers;

import lombok.NonNull;
import org.example.api.dtos.TaskDto;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;
import org.example.api.scheduler.gym.GymEnvironment;

import java.util.List;
import java.util.Optional;

/// A gym workflow mapper.
public interface GymWorkflowMapper<TObservation, TAction> {
    Optional<VmAssignmentDto> schedule(@NonNull GymEnvironment<TObservation, TAction> environment,
                                       @NonNull List<VmDto> vms, @NonNull List<TaskDto> tasks);
}
