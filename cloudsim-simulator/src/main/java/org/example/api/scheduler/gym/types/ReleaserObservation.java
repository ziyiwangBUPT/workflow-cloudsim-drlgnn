package org.example.api.scheduler.gym.types;


import lombok.Builder;

@Builder
public record ReleaserObservation(
        int bufferedTasks,
        int releasedTasks,
        int scheduledTasks,
        int runningTasks,
        int completedTasks,
        int vmCount
) {
}
