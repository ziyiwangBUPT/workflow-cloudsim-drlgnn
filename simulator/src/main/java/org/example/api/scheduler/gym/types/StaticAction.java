package org.example.api.scheduler.gym.types;

import lombok.Data;
import org.example.api.dtos.VmAssignmentDto;

import java.util.List;

@Data
public class StaticAction {
    private final List<VmAssignmentDto> assignments;
}
