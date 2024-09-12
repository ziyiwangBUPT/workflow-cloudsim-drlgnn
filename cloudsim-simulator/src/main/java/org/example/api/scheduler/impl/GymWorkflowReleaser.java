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
import org.example.core.registries.CloudletRegistry;

import java.util.ArrayList;
import java.util.List;

/// The releaser implementation that uses the Gym environment to release workflows.
public class GymWorkflowReleaser implements WorkflowReleaser {
    private final GymEnvironment<ReleaserObservation, ReleaserAction> environment;

    private final List<VmDto> vms = new ArrayList<>();
    private final List<WorkflowDto> workflows = new ArrayList<>();

    public GymWorkflowReleaser(GymSharedQueue<ReleaserObservation, ReleaserAction> queue) {
        this.environment = new GymEnvironment<>(queue);
    }

    @Override
    public void notifyNewVm(@NonNull VmDto newVm) {
        vms.add(newVm);
    }

    @Override
    public void notifyNewWorkflow(@NonNull WorkflowDto newWorkflow) {
        workflows.add(newWorkflow);
    }

    @Override
    public boolean shouldRelease() {
        // Observation
        var cloudletRegistry = CloudletRegistry.getInstance();
        var completionTimeVariance = cloudletRegistry.getCompletionTimeVariance();
        var observation = new ReleaserObservation(workflows.size(), vms.size(), completionTimeVariance);

        // Reward
        var reward = (completionTimeVariance < 0.1) ? 1 : 0;

        // Action
        var action = environment.step(AgentResult.reward(observation, reward));
        var shouldRelease = action.shouldRelease();

        // The state changes if releasing
        if (shouldRelease) workflows.clear();
        return shouldRelease;
    }
}
