package org.example.api.dtos;

import lombok.Data;

@Data
public class VmAssignmentDto {
    private final int vmId;
    private final int vmSubmissionIndex;
    private final int workflowId;
    private final int taskId;
}
