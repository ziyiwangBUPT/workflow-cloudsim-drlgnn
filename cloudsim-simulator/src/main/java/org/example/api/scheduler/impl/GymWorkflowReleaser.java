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
        // Observation is all the vms and workflows
        var observation = new ReleaserObservation(vms, workflows);
        // Reward is negative reward of workflows remaining
        // TODO: Calculate a better reward
        var reward = -workflows.size();

        // Get the action from the agent
        var action = environment.step(AgentResult.reward(observation, reward));
        var shouldRelease = action.shouldRelease();

        // The state changes if releasing
        if (shouldRelease) workflows.clear();
        return shouldRelease;
    }
}
