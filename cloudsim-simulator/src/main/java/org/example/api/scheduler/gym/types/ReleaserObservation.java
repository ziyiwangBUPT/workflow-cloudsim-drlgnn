package org.example.api.scheduler.gym.types;


import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;

import java.util.List;

public record ReleaserObservation(List<VmDto> vms, List<WorkflowDto> workflows) {
}
