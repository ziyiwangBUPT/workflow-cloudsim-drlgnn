package org.example.simulation.listeners;

import lombok.Builder;
import lombok.NonNull;
import org.example.api.scheduler.WorkflowReleaser;
import org.example.api.dtos.WorkflowDto;
import org.example.dataset.DatasetWorkflow;

import java.util.ArrayList;
import java.util.List;

/// Represents the buffer for the workflows.
/// This will hold workflows after they arrive until it gets a signal to release them.
public class WorkflowBuffer extends SimulationTickListener {
    private static final String NAME = "WORKFLOW_BUFFER";

    private final WorkflowReleaser releaser;
    private final WorkflowCoordinator coordinator;
    private final List<DatasetWorkflow> workflowCache = new ArrayList<>();

    @Builder
    public WorkflowBuffer(@NonNull WorkflowReleaser releaser, @NonNull WorkflowCoordinator coordinator) {
        super(NAME);
        this.releaser = releaser;
        this.coordinator = coordinator;
    }

    @Override
    protected void onTick(double time) {
        // No need to continue if there are no workflows
        if (workflowCache.isEmpty()) return;

        // Check if we should release the workflows
        if (releaser.shouldRelease()) {
            coordinator.submitWorkflowsToSystem(workflowCache);
            workflowCache.clear();
        }
    }

    /// Submit a workflow from the user to the buffer.
    public void submitWorkflowFromUser(@NonNull DatasetWorkflow workflow) {
        workflowCache.add(workflow);
        releaser.notifyNewWorkflow(WorkflowDto.from(workflow));
    }
}
