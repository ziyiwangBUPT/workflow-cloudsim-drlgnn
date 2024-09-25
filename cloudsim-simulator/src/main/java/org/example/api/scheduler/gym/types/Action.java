package org.example.api.scheduler.gym.types;

import lombok.Data;

@Data
public class Action {
    private final boolean noOp;
    private final int vmId;
    private final int workflowId;
    private final int taskId;
}
