package org.example.api.dtos;

import lombok.Data;

@Data
public class TaskAssignmentDto {
    private final int workflowId;
    private final int taskId;
    private final int vmId;
}
