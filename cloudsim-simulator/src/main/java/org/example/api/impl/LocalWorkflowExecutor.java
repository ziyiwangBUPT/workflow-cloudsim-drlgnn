package org.example.api.impl;

import lombok.NonNull;
import org.example.api.WorkflowExecutor;
import org.example.api.dtos.*;
import org.example.utils.DependencyCountMap;

import java.util.*;

/// A CloudSim implemented local workflow executor.
public class LocalWorkflowExecutor implements WorkflowExecutor {
    private final Map<WorkflowTaskId, TaskDto> taskMap = new HashMap<>();
    private final Map<WorkflowTaskId, VmId> scheduledMap = new HashMap<>();
    private final Set<WorkflowTaskId> executingTasks = new HashSet<>();
    private final DependencyCountMap<WorkflowTaskId> pendingDependencies = new DependencyCountMap<>();

    private final Map<VmId, Queue<WorkflowTaskId>> pendingTaskQueueByVm = new HashMap<>(); // Tasks that are pending for VM
    private final Map<VmId, WorkflowTaskId> readyTaskByVm = new HashMap<>(); // Tasks that are ready to be executed in VM

    @Override
    public void notifyNewVm(@NonNull VmDto newVm) {
        pendingTaskQueueByVm.put(new VmId(newVm.getId()), new LinkedList<>());
    }

    @Override
    public void notifyNewWorkflow(@NonNull WorkflowDto newWorkflow) {
        // Submit to map for tasks for easy access.
        // Also create a map for holding number of pending dependencies for each task.
        for (var task : newWorkflow.getTasks()) {
            var taskId = new WorkflowTaskId(newWorkflow.getId(), task.getId());
            taskMap.put(taskId, task);

            // Add dependency from parent task
            for (var childId : task.getChildIds()) {
                var childTaskId = new WorkflowTaskId(newWorkflow.getId(), childId);
                pendingDependencies.addNewDependency(childTaskId);
            }
        }
    }

    @Override
    public void notifyScheduling(@NonNull VmAssignmentDto assignment) {
        var vmId = new VmId(assignment.getVmId());
        var workflowTaskId = new WorkflowTaskId(assignment.getWorkflowId(), assignment.getTaskId());
        scheduledMap.put(workflowTaskId, vmId);

        // Move the task to the pending queue
        var pendingQueue = pendingTaskQueueByVm.get(vmId);
        pendingQueue.add(workflowTaskId);

        updateReadyTasks();
    }

    @Override
    public void notifyCompletion(int workflowId, int taskId) {
        var workflowTaskId = new WorkflowTaskId(workflowId, taskId);
        var vmId = scheduledMap.get(workflowTaskId);

        // Update tasks that were dependent on the task that just completed
        var task = taskMap.get(workflowTaskId);
        for (var childId : task.getChildIds()) {
            var childTaskId = new WorkflowTaskId(workflowId, childId);
            pendingDependencies.removeOneDependency(childTaskId);
        }

        // Remove the task from the executing list
        readyTaskByVm.remove(vmId);
        executingTasks.remove(workflowTaskId);

        updateReadyTasks();
    }

    @Override
    public List<TaskAssignmentDto> pollTaskAssignments() {
        var assignments = new ArrayList<TaskAssignmentDto>();

        // Find the VMs that are ready to execute
        for (var vmId : readyTaskByVm.keySet()) {
            var taskId = readyTaskByVm.get(vmId);
            if (!executingTasks.contains(taskId)) {
                executingTasks.add(taskId);
                var assignment = new TaskAssignmentDto(taskId.workflowId(), taskId.taskId(), vmId.vmId());
                assignments.add(assignment);
            }
        }

        return assignments;
    }

    /// Updates the ready list based on the pending queues and dependencies.
    /// Only needs to call if the ready list changes or dependencies get removed.
    private void updateReadyTasks() {
        // Only the ones at the front of the queue are ready
        for (var vmId : pendingTaskQueueByVm.keySet()) {
            var queue = pendingTaskQueueByVm.get(vmId);
            if (!queue.isEmpty() && !readyTaskByVm.containsKey(vmId)) {
                var readyTaskId = queue.peek();
                if (pendingDependencies.hasNoDependency(readyTaskId)) {
                    queue.poll();
                    readyTaskByVm.put(vmId, readyTaskId);
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
