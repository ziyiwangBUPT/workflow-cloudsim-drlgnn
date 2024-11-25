package org.example.api.scheduler.gym.types;


import lombok.Data;
import org.example.api.dtos.TaskDto;
import org.example.api.dtos.VmDto;

import java.util.List;

@Data
public class StaticObservation {
    private final List<TaskDto> tasks;
    private final List<VmDto> vms;
}
