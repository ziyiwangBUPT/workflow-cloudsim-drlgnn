package org.example.api.impl;

import lombok.NonNull;
import org.example.api.WorkflowExecutor;
import org.example.api.dtos.TaskAssignmentDto;
import org.example.api.dtos.TaskDto;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.WorkflowDto;
import org.example.utils.DependencyCountMap;

import java.util.*;

/// A CloudSim implemented local workflow executor.
public class LocalWorkflowExecutor implements WorkflowExecutor {
    private final Map<WorkflowTaskId, TaskDto> taskMap = new HashMap<>();
    private final Map<WorkflowTaskId, VmId> assignedVmMap = new HashMap<>();
    private final DependencyCountMap<WorkflowTaskId> pendingDependencies = new DependencyCountMap<>();

    private final Map<VmId, Queue<WorkflowTaskId>> pendingQueues = new HashMap<>();
    private final List<WorkflowTaskId> readyList = new ArrayList<>();

    @Override
    public void submitAssignments(@NonNull List<WorkflowDto> workflows, List<VmAssignmentDto> assignments) {
        // First, sort the assignments by the submission index.
        // This is to make sure that the tasks are executed in the order they were submitted in VMs.
        assignments.sort(Comparator.comparingInt(VmAssignmentDto::getVmSubmissionIndex));

        // Simply add the assignments to the VM queues.
        for (var assignment : assignments) {
            var vmId = new VmId(assignment.getVmId());
            var workflowTaskId = new WorkflowTaskId(assignment.getWorkflowId(), assignment.getTaskId());
            var queue = pendingQueues.computeIfAbsent(vmId, _ -> new ArrayDeque<>());
            assignedVmMap.put(workflowTaskId, vmId);
            queue.add(workflowTaskId);
        }

        // Submit to map for tasks for easy access.
        // Also create a map for holding number of pending dependencies for each task.
        for (var workflow : workflows) {
            for (var task : workflow.getTasks()) {
                var taskId = new WorkflowTaskId(workflow.getId(), task.getId());
                taskMap.put(taskId, task);
                var queue = pendingQueues.get(assignedVmMap.get(taskId));
                if (!taskId.equals(queue.peek())) {
                    // This is not at the start of the queue - dependency from VM prev task
                    pendingDependencies.addNewDependency(taskId);
                }
                for (var childId : task.getChildIds()) {
                    // Dependency from parent task
                    var childTaskId = new WorkflowTaskId(workflow.getId(), childId);
                    pendingDependencies.addNewDependency(childTaskId);
                }
            }
        }

        updateReadyList();
    }

    @Override
    public void notifyCompletion(int workflowId, int taskId) {
        var workflowTaskId = new WorkflowTaskId(workflowId, taskId);
        var vmId = assignedVmMap.remove(workflowTaskId);

        // There are 2 types of contenders to update:
        // (1) The next task in the queue of the VM that just completed the task
        var queue = pendingQueues.get(vmId);
        var nextTaskInQueue = queue.peek();
        if (nextTaskInQueue != null) {
            // Dependency from prev task
            pendingDependencies.removeOneDependency(nextTaskInQueue);
        }
        // (2) The tasks that were dependent on the task that just completed
        var task = taskMap.get(workflowTaskId);
        for (var childId : task.getChildIds()) {
            // Dependency from parent task
            var childTaskId = new WorkflowTaskId(workflowId, childId);
            pendingDependencies.removeOneDependency(childTaskId);
        }

        updateReadyList();
    }

    @Override
    public List<TaskAssignmentDto> pollTaskAssignments() {
        var assignments = new ArrayList<TaskAssignmentDto>();
        for (var readyTaskId : readyList) {
            var assignedVmId = assignedVmMap.get(readyTaskId);
            var assignment = new TaskAssignmentDto(readyTaskId.workflowId(), readyTaskId.taskId(), assignedVmId.vmId());
            assignments.add(assignment);
        }
        readyList.clear();

        return assignments;
    }

    /// Updates the ready list based on the pending queues and dependencies.
    private void updateReadyList() {
        // Only the ones at the front of the queue are ready
        for (var queue : pendingQueues.values()) {
            if (!queue.isEmpty()) {
                var readyTaskId = queue.peek();
                if (pendingDependencies.hasNoDependency(readyTaskId)) {
                    queue.poll();
                    readyList.add(readyTaskId);
                }
            }
        }
    }

    /// A private record to represent a VM ID.
    private record VmId(int vmId) {
    }

    /// A private record to represent a workflow task ID.
    private record WorkflowTaskId(int workflowId, int taskId) {
    }
}
