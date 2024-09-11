package org.example.ticks;

import lombok.Builder;
import lombok.NonNull;
import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.Vm;
import org.example.api.WorkflowExecutor;
import org.example.api.WorkflowReleaser;
import org.example.api.WorkflowScheduler;
import org.example.api.dtos.HostDto;
import org.example.api.dtos.TaskDto;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;
import org.example.dataset.DatasetTask;
import org.example.dataset.DatasetWorkflow;
import org.example.entities.DynamicDatacenterBroker;
import org.example.entities.MonitoredHost;
import org.example.factories.CloudletFactory;

import java.util.*;
import java.util.stream.Collectors;

/// The coordinator is responsible for coordinating the simulation.
/// It listens to the simulation ticks and performs the following tasks:
/// 1. Discover new VMs from the broker and submit them to the releaser.
/// 2. Consume workflows from the user and submit them to the releaser.
/// 3. Discover completed cloudlets from the broker and notify the executor.
/// 4. Check with the releaser on whether to release workflows.
/// 5. Poll the executor for new ready tasks and create cloudlets for them.
public class Coordinator extends SimulationTickListener {
    private static final String NAME = "COORDINATOR";

    private final DynamicDatacenterBroker broker;
    private final CloudletFactory cloudletFactory;

    private final WorkflowReleaser releaser;
    private final WorkflowScheduler scheduler;
    private final WorkflowExecutor executor;

    private final Map<VmId, VmDto> vmMap = new HashMap<>();
    private final Map<WorkflowTaskId, DatasetTask> taskMap = new HashMap<>();
    private final Queue<DatasetWorkflow> userWorkflowBacklog = new ArrayDeque<>(); // Unsubmitted
    private final List<WorkflowDto> workflowCache = new ArrayList<>(); // Submitted but unreleased
    private final Map<WorkflowTaskId, Cloudlet> executingCloudlets = new HashMap<>(); // Submitted but not finished

    @Builder
    public Coordinator(@NonNull WorkflowReleaser releaser,
                       @NonNull WorkflowScheduler scheduler,
                       @NonNull WorkflowExecutor executor,
                       @NonNull DynamicDatacenterBroker broker,
                       @NonNull CloudletFactory cloudletFactory,
                       @NonNull List<DatasetWorkflow> workflows) {
        super(NAME);

        this.releaser = releaser;
        this.scheduler = scheduler;
        this.executor = executor;
        this.broker = broker;
        this.cloudletFactory = cloudletFactory;

        // Sort the pending workflows by arrival time (ascending) and add to the queue
        workflows.stream().sorted(Comparator.comparingInt(DatasetWorkflow::getArrivalTime))
                .forEach(userWorkflowBacklog::add);
    }

    @Override
    protected void onTick(double time) {
        discoverNewVmsFromCloudSim();
        consumeWorkflowsFromUser(time);
        discoverCompletedCloudletsFromCloudSim();

        releaseWorkflows();
        submitAndExecuteCloudlets();
    }

    /// Check for the broker's VM list and discover any new VMs.
    /// New VMs are submitted to the releaser.
    private void discoverNewVmsFromCloudSim() {
        // No need to continue if there are no new VMs
        if (broker.getGuestsCreatedList().isEmpty()) return;
        if (broker.getGuestsCreatedList().size() == vmMap.size()) return;

        // Filter out the VMs that are already submitted
        var newVms = new HashMap<VmId, VmDto>();
        for (var vm : broker.getGuestsCreatedList()) {
            var vmId = new VmId(vm.getId());
            if (vmMap.containsKey(vmId)) continue;
            newVms.put(vmId, toDto((Vm) vm));
        }

        vmMap.putAll(newVms);
        releaser.submitVms(newVms.values().stream().toList());
    }

    /// Check for the unsubmitted workflows and submit them if they have arrived.
    /// Submitted tasks are added to the unreleased list as well as the releaser.
    private void consumeWorkflowsFromUser(double currentTime) {
        // No need to continue if there are no workflows
        if (userWorkflowBacklog.isEmpty()) return;

        // Only consume workflows that have arrived
        var newWorkflows = new ArrayList<WorkflowDto>();
        var newTasks = new HashMap<WorkflowTaskId, DatasetTask>();
        while (!userWorkflowBacklog.isEmpty()) {
            var arrivalTime = userWorkflowBacklog.peek().getArrivalTime();
            if (arrivalTime > currentTime) {
                break;
            }

            // Workflow arrived, submit it
            var workflow = userWorkflowBacklog.poll();
            newWorkflows.add(toDto(workflow));
            for (var task : workflow.getTasks()) {
                newTasks.put(new WorkflowTaskId(workflow.getId(), task.getId()), task);
            }
        }

        taskMap.putAll(newTasks);
        workflowCache.addAll(newWorkflows);
        releaser.submitWorkflows(newWorkflows);
    }

    /// Check for finished cloudlets and notify the executor.
    private void discoverCompletedCloudletsFromCloudSim() {
        var finishedWorkflowTaskIds = new ArrayList<WorkflowTaskId>();

        // Check for finished cloudlets
        for (var workflowTaskId : executingCloudlets.keySet()) {
            var cloudlet = executingCloudlets.get(workflowTaskId);
            if (cloudlet.isFinished()) {
                executor.notifyCompletion(workflowTaskId.workflowId(), workflowTaskId.taskId());
                finishedWorkflowTaskIds.add(workflowTaskId);
            }
        }

        // Remove the finished cloudlets
        finishedWorkflowTaskIds.forEach(executingCloudlets::remove);
    }

    /// Check with releaser on whether to release.
    /// If releaser says yes, schedule the workflows.
    private void releaseWorkflows() {
        if (releaser.shouldRelease()) {
            var assignments = scheduler.schedule(workflowCache, vmMap.values().stream().toList());
            executor.submitAssignments(workflowCache, assignments);
            workflowCache.clear();
        }
    }

    /// Poll the executor for new ready tasks and create cloudlets for them.
    /// Also submit and schedule the created cloudlets.
    private void submitAndExecuteCloudlets() {
        var newCloudlets = new HashMap<WorkflowTaskId, Cloudlet>();

        // Create cloudlets for the assigned tasks
        var taskAssignments = executor.pollTaskAssignments();
        for (var assignment : taskAssignments) {
            var workflowTaskId = new WorkflowTaskId(assignment.getWorkflowId(), assignment.getTaskId());
            var task = taskMap.get(workflowTaskId);
            var cloudlet = cloudletFactory.createCloudlet(broker.getId(), task);
            cloudlet.setGuestId(assignment.getVmId());
            newCloudlets.put(workflowTaskId, cloudlet);
        }

        // Submit and schedule the cloudlets
        broker.submitCloudletList(newCloudlets.values().stream().toList());
        broker.scheduleSubmittedCloudlets();
        executingCloudlets.putAll(newCloudlets);
    }

    // Helper methods to convert entities to DTOs --------------------------------------------

    private VmDto toDto(@NonNull Vm vm) {
        var host = (MonitoredHost) vm.getHost();
        return VmDto.builder()
                .id(vm.getId())
                .host(HostDto.builder()
                        .id(host.getId())
                        .cores(host.getNumberOfPes())
                        .cpuSpeedMips(host.getTotalMips())
                        .memoryMb(host.getRam())
                        .diskMb(host.getStorage())
                        .bandwidthMbps(host.getBw())
                        .powerIdleWatt(host.getPowerModel().getPower(0))
                        .powerPeakWatt(host.getPowerModel().getPower(1))
                        .build())
                .cores(vm.getNumberOfPes())
                .cpuSpeedMips(vm.getMips())
                .memoryMb(vm.getRam())
                .diskMb(vm.getSize())
                .bandwidthMbps(vm.getBw())
                .vmm(vm.getVmm())
                .build();
    }

    private WorkflowDto toDto(@NonNull DatasetWorkflow workflow) {
        return WorkflowDto.builder()
                .id(workflow.getId())
                .tasks(workflow.getTasks().stream().map(
                        task -> TaskDto.builder()
                                .id(task.getId())
                                .length(task.getLength())
                                .reqCores(task.getReqCores())
                                .childIds(task.getChildIds())
                                .build()
                ).collect(Collectors.toList()))
                .build();
    }

    /// A private record to represent a VM ID.
    private record VmId(int vmId) {
    }

    /// A private record to represent a workflow task ID.
    private record WorkflowTaskId(int workflowId, int taskId) {
    }
}
