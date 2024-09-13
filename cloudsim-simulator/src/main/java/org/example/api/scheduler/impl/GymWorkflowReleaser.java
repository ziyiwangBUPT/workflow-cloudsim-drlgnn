package org.example.api.scheduler.impl;

import lombok.NonNull;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;
import org.example.api.scheduler.WorkflowReleaser;
import org.example.api.scheduler.gym.GymEnvironment;
import org.example.api.scheduler.gym.GymSharedQueue;
import org.example.api.scheduler.gym.types.AgentResult;
import org.example.api.scheduler.gym.types.ReleaserAction;
import org.example.api.scheduler.gym.types.ReleaserObservation;
import org.example.sensors.TaskStateSensor;

import java.util.ArrayList;
import java.util.List;

/// The releaser implementation that uses the Gym environment to release workflows.
public class GymWorkflowReleaser implements WorkflowReleaser {
    private final GymEnvironment<ReleaserObservation, ReleaserAction> environment;

    private final List<VmDto> vms = new ArrayList<>();

    public GymWorkflowReleaser(GymSharedQueue<ReleaserObservation, ReleaserAction> queue) {
        this.environment = new GymEnvironment<>(queue);
    }

    @Override
    public void notifyNewVm(@NonNull VmDto newVm) {
        vms.add(newVm);
    }

    @Override
    public void notifyNewWorkflow(@NonNull WorkflowDto newWorkflow) {
    }

    @Override
    public boolean shouldRelease() {
        var taskStateSensor = TaskStateSensor.getInstance();

        // Observation
        var observation = ReleaserObservation.builder()
                .bufferedTasks(taskStateSensor.getBufferedTasks())
                .releasedTasks(taskStateSensor.getReleasedTasks())
                .scheduledTasks(taskStateSensor.getScheduledTasks())
                .runningTasks(taskStateSensor.getExecutedTasks())
                .completedTasks(taskStateSensor.getCompletedTasks())
                .vmCount(vms.size()).build();

        // Reward
        var tasksInRunning = taskStateSensor.getExecutedTasks() - taskStateSensor.getCompletedTasks();
        var positiveReward = (vms.size() - tasksInRunning) / (double) vms.size();
        var negativeReward = (taskStateSensor.getBufferedTasks() - taskStateSensor.getReleasedTasks()) / (double) taskStateSensor.getBufferedTasks();
        var reward = positiveReward - 2 * negativeReward;

        // Action
        var action = environment.step(AgentResult.reward(observation, reward));
        return action.shouldRelease();
    }
}
