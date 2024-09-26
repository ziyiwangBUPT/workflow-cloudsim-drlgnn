package org.example.api.scheduler.gym;

import lombok.Setter;
import lombok.experimental.Accessors;
import org.example.api.scheduler.WorkflowScheduler;
import org.example.api.scheduler.gym.algorithms.StaticGymSchedulingAlgorithm;
import org.example.api.scheduler.gym.types.StaticAction;
import org.example.api.scheduler.gym.types.StaticObservation;
import org.example.api.scheduler.internal.StaticWorkflowScheduler;
import org.example.api.scheduler.internal.algorithms.RoundRobinSchedulingAlgorithm;

/// Factory for creating WorkflowScheduler instances.
@Accessors(chain = true, fluent = true)
@Setter
public class WorkflowSchedulerFactory {
    private GymSharedQueue<StaticObservation, StaticAction> staticSharedQueue;

    /// Create a new WorkflowScheduler instance.
    public WorkflowScheduler create(String algorithm) {
        return switch (algorithm) {
            case "switch:gym" -> new StaticWorkflowScheduler(new StaticGymSchedulingAlgorithm(staticSharedQueue));
            case "static:round-robin" -> new StaticWorkflowScheduler(new RoundRobinSchedulingAlgorithm());
            default -> throw new IllegalArgumentException("Invalid algorithm: " + algorithm);
        };
    }
}
