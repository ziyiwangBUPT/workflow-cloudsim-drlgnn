package org.example.sensors;

import lombok.Getter;
import org.example.core.registries.CloudletRegistry;

public class RewardSensor {
    private static final double MAKESPAN_W = 1;
    private static final double INCOMPLETE_W = 5;

    @Getter
    private static final RewardSensor instance = new RewardSensor();

    private RewardSensor() {
    }

    public double finalReward(double totalDuration) {
        var cloudletRegistry = CloudletRegistry.getInstance();
        var taskStateSensor = TaskStateSensor.getInstance();

        // More the makespan, more the penalty (0 <= makespanPenalty <= 1)
        var makespanPenalty = cloudletRegistry.getMakespan() / totalDuration;
        // More tasks are incomplete, more the penalty (0 <= unscheduledPenalty <= 1)
        var incompletePenalty = taskStateSensor.getIncompleteTasks() / (double) taskStateSensor.getBufferedTasks();
        // Total penalty (0 <= penalty <= 1)
        var totalPenalty = (MAKESPAN_W * makespanPenalty + INCOMPLETE_W * incompletePenalty) / (MAKESPAN_W + INCOMPLETE_W);

        return -totalPenalty;
    }
}
