package org.example.dataset;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class DatasetVmAssignment {
    private final int workflowId;
    private final int taskId;
    private final int vmId;
    private final double startTime;
    private final double endTime;
}
