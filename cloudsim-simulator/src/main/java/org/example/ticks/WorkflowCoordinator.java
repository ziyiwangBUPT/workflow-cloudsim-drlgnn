package org.example.ticks;

import lombok.Builder;
import lombok.NonNull;
import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.Vm;
import org.example.api.WorkflowExecutor;
import org.example.api.WorkflowReleaser;
import org.example.api.WorkflowScheduler;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;
import org.example.dataset.DatasetTask;
import org.example.dataset.DatasetWorkflow;
import org.example.entities.DynamicDatacenterBroker;
import org.example.factories.CloudletFactory;

import java.util.*;

/// The coordinator is responsible for coordinating the simulation.
public class WorkflowCoordinator extends SimulationTickListener {
    private static final String NAME = "COORDINATOR";

    private final DynamicDatacenterBroker broker;
    private final CloudletFactory cloudletFactory;

    private final WorkflowReleaser releaser;
    private final WorkflowScheduler scheduler;
    private final WorkflowExecutor executor;

    private final Set<VmId> discoveredVms = new HashSet<>();
    private final Map<WorkflowTaskId, DatasetTask> unscheduledTasks = new HashMap<>();
    private final Map<WorkflowTaskId, Cloudlet> executingTasks = new HashMap<>();

    @Builder
    public WorkflowCoordinator(@NonNull WorkflowReleaser releaser,
                               @NonNull WorkflowScheduler scheduler,
                               @NonNull WorkflowExecutor executor,
                               @NonNull DynamicDatacenterBroker broker,
                               @NonNull CloudletFactory cloudletFactory) {
        super(NAME);

        this.releaser = releaser;
        this.scheduler = scheduler;
        this.executor = executor;
        this.broker = broker;
        this.cloudletFactory = cloudletFactory;
    }

    @Override
    protected void onTick(double time) {
        discoverNewVmsFromCloudSim();
        discoverCompletedCloudletsFromCloudSim();

        // Schedule tasks to VMs
        var scheduling = scheduler.schedule();
        while (scheduling.isPresent()) {
            executor.notifyScheduling(scheduling.get());
            scheduling = scheduler.schedule();
        }

        submitAndExecuteCloudlets();
    }

    /// Submit workflows to the system.
    /// This is called by the workflow buffer when it decides to release workflows.
    public void submitWorkflowsToSystem(@NonNull List<DatasetWorkflow> workflows) {
        for (var workflow : workflows) {
            var workflowDto = WorkflowDto.from(workflow);
            scheduler.notifyNewWorkflow(workflowDto);
            executor.notifyNewWorkflow(workflowDto);
            for (var task : workflow.getTasks()) {
                unscheduledTasks.put(new WorkflowTaskId(workflow.getId(), task.getId()), task);
            }
        }
    }

    /// Check for the broker's VM list and discover any new VMs.
    /// New VMs are submitted to the releaser.
    private void discoverNewVmsFromCloudSim() {
        // No need to continue if there are no new VMs
        if (broker.getGuestsCreatedList().isEmpty()) return;
        if (broker.getGuestsCreatedList().size() == discoveredVms.size()) return;

        // Filter out the VMs that are already submitted
        for (var vm : broker.getGuestsCreatedList()) {
            var vmId = new VmId(vm.getId());
            if (discoveredVms.contains(vmId)) continue;
            var vmDto = VmDto.from((Vm) vm);
            releaser.notifyNewVm(vmDto);
            scheduler.notifyNewVm(vmDto);
            executor.notifyNewVm(vmDto);
            discoveredVms.add(vmId);
        }
    }

    /// Check for finished cloudlets and notify the executor.
    private void discoverCompletedCloudletsFromCloudSim() {
        // Check for finished cloudlets
        var finishedTasks = new ArrayList<WorkflowTaskId>();
        for (var workflowTaskId : executingTasks.keySet()) {
            var cloudlet = executingTasks.get(workflowTaskId);
            if (cloudlet.isFinished()) {
                finishedTasks.add(workflowTaskId);
            }
        }

        // Removing later to avoid concurrent modification
        for (var workflowTaskId : finishedTasks) {
            executor.notifyCompletion(workflowTaskId.workflowId(), workflowTaskId.taskId());
            executingTasks.remove(workflowTaskId);
        }
    }

    /// Poll the executor for new ready tasks and create cloudlets for them.
    /// Also submit and schedule the created cloudlets.
    private void submitAndExecuteCloudlets() {
        // No need to continue if there are no new task assignments
        var taskAssignments = executor.pollTaskAssignments();
        if (taskAssignments.isEmpty()) return;

        // Create cloudlets for the assigned tasks
        var newCloudlets = new HashMap<WorkflowTaskId, Cloudlet>();
        for (var assignment : taskAssignments) {
            var workflowTaskId = new WorkflowTaskId(assignment.getWorkflowId(), assignment.getTaskId());
            var task = unscheduledTasks.remove(workflowTaskId);
            var cloudlet = cloudletFactory.createCloudlet(broker.getId(), task);
            cloudlet.setGuestId(assignment.getVmId());
            newCloudlets.put(workflowTaskId, cloudlet);
        }

        // Submit and schedule the cloudlets
        broker.submitCloudletList(newCloudlets.values().stream().toList());
        broker.scheduleSubmittedCloudlets();
        executingTasks.putAll(newCloudlets);
    }

    /// A private record to represent a VM ID.
    private record VmId(int vmId) {
    }

    /// A private record to represent a workflow task ID.
    private record WorkflowTaskId(int workflowId, int taskId) {
    }
}
