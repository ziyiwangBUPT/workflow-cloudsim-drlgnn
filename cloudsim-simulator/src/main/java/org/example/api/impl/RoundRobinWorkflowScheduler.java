package org.example.api.impl;

import lombok.NonNull;
import org.example.api.WorkflowScheduler;
import org.example.api.dtos.TaskDto;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;
import org.example.utils.DependencyCountMap;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/// Round-robin workflow scheduler implementation.
public class RoundRobinWorkflowScheduler implements WorkflowScheduler {
    private final List<VmDto> vms = new ArrayList<>();

    private final Map<WorkflowTaskId, TaskDto> taskMap = new HashMap<>();
    private final DependencyCountMap<WorkflowTaskId> pendingDependencies = new DependencyCountMap<>();
    private final Queue<WorkflowTaskId> readyQueue = new LinkedList<>();

    private final AtomicInteger index = new AtomicInteger(0);

    @Override
    public void notifyNewVm(@NonNull VmDto newVm) {
        vms.add(newVm);
    }

    @Override
    public void notifyNewWorkflow(@NonNull WorkflowDto newWorkflow) {
        // Update the task map
        for (var task : newWorkflow.getTasks()) {
            var taskId = new WorkflowTaskId(newWorkflow.getId(), task.getId());
            taskMap.put(taskId, task);
        }

        // Update the pending dependencies map with new tasks
        for (var task : newWorkflow.getTasks()) {
            for (var childId : task.getChildIds()) {
                var childTaskId = new WorkflowTaskId(newWorkflow.getId(), childId);
                pendingDependencies.addNewDependency(childTaskId);
            }
        }

        // The start node of the workflow is ready to be scheduled
        readyQueue.add(new WorkflowTaskId(newWorkflow.getId(), 0));
    }

    @Override
    public Optional<VmAssignmentDto> schedule() {
        if (readyQueue.isEmpty()) return Optional.empty();

        var workflowTaskId = readyQueue.poll();
        var task = taskMap.get(workflowTaskId);

        // Find the next best VM that is suitable for the task
        var bestVm = vms.get(index.getAndIncrement() % vms.size());
        while (bestVm.getCores() < task.getReqCores()) {
            bestVm = vms.get(index.getAndIncrement() % vms.size());
        }

        var vmAssignment = new VmAssignmentDto(bestVm.getId(), task.getWorkflowId(), task.getId());

        // Add the task to the ready set if all dependencies are done
        // Only process child tasks since they are the ones becoming ready
        for (var childId : task.getChildIds()) {
            var childTaskId = new WorkflowTaskId(task.getWorkflowId(), childId);
            pendingDependencies.removeOneDependency(childTaskId);
            if (pendingDependencies.hasNoDependency(childTaskId)) {
                // This was the last dependency
                readyQueue.add(childTaskId);
            }
        }

        return Optional.of(vmAssignment);
    }

    /// A private record to represent a workflow task ID.
    private record WorkflowTaskId(int workflowId, int taskId) {
    }
}
