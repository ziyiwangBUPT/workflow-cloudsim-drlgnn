package org.example.api.dtos;

import lombok.Builder;
import lombok.Data;

import java.util.List;

@Data
@Builder
public class WorkflowDto {
    private final int id;
    private final List<TaskDto> tasks;
}
