package org.example.ticks;

import lombok.Builder;
import org.example.entities.ExecutionPlan;
import org.example.entities.MonitoredDatacenterBroker;

public class WorkflowExecutor extends SimulationTickListener {
    private static final String NAME = "WORKFLOW_EXECUTOR";

    private final MonitoredDatacenterBroker broker;
    private final ExecutionPlan executionPlan;

    @Builder
    protected WorkflowExecutor(MonitoredDatacenterBroker broker, ExecutionPlan executionPlan) {
        super(NAME);
        this.broker = broker;
        this.executionPlan = executionPlan;
    }

    @Override
    protected void onTick(double timeMs) {
        executionPlan.freeFinishedTasks();
        executionPlan.submitReadyTasks(broker);
        broker.scheduleSubmittedCloudlets();
    }
}
