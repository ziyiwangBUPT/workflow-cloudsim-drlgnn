package org.example.ticks;

import lombok.Builder;
import org.cloudbus.cloudsim.Vm;
import org.example.entities.ExecutionPlan;
import org.example.factories.CloudletFactory;
import org.example.dataset.DatasetWorkflow;
import org.example.entities.DynamicDatacenterBroker;
import org.example.schedulers.Scheduler;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;

import static java.util.Comparator.*;

public class WorkflowScheduler extends SimulationTickListener {
    private static final String NAME = "WORKFLOW_SCHEDULER";

    private final long schedulingIntervalMs;
    private final DynamicDatacenterBroker broker;
    private final Queue<DatasetWorkflow> pendingWorkflows;
    private final CloudletFactory cloudletFactory;
    private final Scheduler scheduler;
    private final ExecutionPlan executionPlan;

    private final List<DatasetWorkflow> arrivedWorkflows = new ArrayList<>();
    private double nextScheduleAtMs = 0;

    @Builder
    protected WorkflowScheduler(DynamicDatacenterBroker broker,
                                Scheduler scheduler,
                                ExecutionPlan executionPlan,
                                CloudletFactory cloudletFactory,
                                long schedulingInterval,
                                List<DatasetWorkflow> datasetWorkflows) {
        super(NAME);
        this.broker = broker;
        this.scheduler = scheduler;
        this.executionPlan = executionPlan;
        this.cloudletFactory = cloudletFactory;
        this.schedulingIntervalMs = schedulingInterval;

        // Sort the pending cloudlets by arrival time (ascending)
        datasetWorkflows.sort(comparingDouble(DatasetWorkflow::getArrivalTime));
        this.pendingWorkflows = new ArrayDeque<>(datasetWorkflows);
    }

    @Override
    protected void onTick(double timeMs) {
        // Update arrived workflows
        while (!pendingWorkflows.isEmpty()) {
            var arrivalTimeS = pendingWorkflows.peek().getArrivalTime();
            if (arrivalTimeS * 1000 > timeMs) {
                break;
            }
            var workflow = pendingWorkflows.remove();
            arrivedWorkflows.add(workflow);
        }

        // Schedule cloudlets periodically and update the execution plan
        if (nextScheduleAtMs <= timeMs && !arrivedWorkflows.isEmpty()) {
            nextScheduleAtMs += schedulingIntervalMs;
            List<Vm> vms = broker.getGuestsCreatedList();
            var cloudlets = cloudletFactory.createCloudlets(broker.getId(), arrivedWorkflows);
            var newExecutionPlan = scheduler.schedule(cloudlets, vms);
            executionPlan.merge(newExecutionPlan);
            arrivedWorkflows.clear();
        }
    }
}
