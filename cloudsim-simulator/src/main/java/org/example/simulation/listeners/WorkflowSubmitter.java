package org.example.simulation.listeners;

import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import org.example.dataset.DatasetWorkflow;

import java.util.*;

/// Represents a user that submits workflows to the system.
/// Uses a dataset of workflows and submits them to the system when they arrive.
public class WorkflowSubmitter extends SimulationTickListener {
    private static final String NAME = "WORKFLOW_SUBMITTER";

    private final WorkflowCoordinator coordinator;
    private final Queue<DatasetWorkflow> workflowBacklog = new LinkedList<>();
    @Getter
    private final List<DatasetWorkflow> submittedWorkflows = new ArrayList<>();

    @Builder
    public WorkflowSubmitter(@NonNull List<DatasetWorkflow> workflows, @NonNull WorkflowCoordinator coordinator) {
        super(NAME);
        this.coordinator = coordinator;

        // Sort the pending workflows by arrival time (ascending) and add to the queue
        workflows.stream().sorted(Comparator.comparingInt(DatasetWorkflow::getArrivalTime))
                .forEach(workflowBacklog::add);
    }

    @Override
    protected void onTick(double time) {
        // No need to continue if there are no workflows
        if (workflowBacklog.isEmpty()) return;

        // Only consume workflows that have arrived
        while (!workflowBacklog.isEmpty()) {
            var arrivalTime = workflowBacklog.peek().getArrivalTime();
            if (arrivalTime > time) {
                break;
            }

            // Workflow arrived, submit it
            var workflow = workflowBacklog.poll();
            coordinator.submitWorkflowFromUser(workflow);
            submittedWorkflows.add(workflow);
        }
    }
}
