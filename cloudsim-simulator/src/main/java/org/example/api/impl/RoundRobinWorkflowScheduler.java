package org.example.api.impl;

import lombok.NonNull;
import org.example.api.WorkflowScheduler;
import org.example.api.dtos.TaskDto;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;
import org.example.utils.DependencyCountMap;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/// Round-robin workflow scheduler implementation.
public class RoundRobinWorkflowScheduler implements WorkflowScheduler {
    @Override
    public List<VmAssignmentDto> schedule(@NonNull List<WorkflowDto> workflows, @NonNull List<VmDto> vms) {
        // Create maps for workflows and tasks for easy access
        var workflowMap = new HashMap<WorkflowId, WorkflowDto>();
        var taskMap = new HashMap<WorkflowTaskId, TaskDto>();
        for (var workflow : workflows) {
            workflowMap.put(new WorkflowId(workflow.getId()), workflow);
            for (var task : workflow.getTasks()) {
                taskMap.put(new WorkflowTaskId(workflow.getId(), task.getId()), task);
            }
        }

        // Create map for holding number of pending dependencies for each task
        var pendingDependencies = new DependencyCountMap<WorkflowTaskId>();
        for (var workflow : workflows) {
            for (var task : workflow.getTasks()) {
                for (var childId : task.getChildIds()) {
                    var childTaskId = new WorkflowTaskId(workflow.getId(), childId);
                    pendingDependencies.addNewDependency(childTaskId);
                }
            }
        }

        // Add the start task of each workflow to the ready queue
        var ready = new ArrayDeque<WorkflowTaskId>();
        for (var workflow : workflows) {
            ready.add(new WorkflowTaskId(workflow.getId(), 0));
        }

        // Data structures for holding the VM assignments and counts
        var vmAssignmentMap = new HashMap<WorkflowTaskId, VmAssignmentDto>();
        var vmAssignedCountMap = new HashMap<VmId, Integer>();

        var index = 0;
        while (!ready.isEmpty()) {
            var workflowTaskId = ready.removeFirst();
            var workflow = workflowMap.get(new WorkflowId(workflowTaskId.workflowId()));
            var task = taskMap.get(workflowTaskId);

            // Find the next best VM that is suitable for the task
            var bestVm = vms.get((index++) % vms.size());
            while (bestVm.getCores() < task.getReqCores()) {
                bestVm = vms.get((index++) % vms.size());
            }

            // Submit the task to the VM
            var bestId = new VmId(bestVm.getId());
            var vmSubmissionId = vmAssignedCountMap.getOrDefault(bestId, 0);
            var vmAssignment = new VmAssignmentDto(bestVm.getId(), vmSubmissionId, workflow.getId(), task.getId());
            vmAssignedCountMap.put(bestId, vmSubmissionId + 1);
            vmAssignmentMap.put(workflowTaskId, vmAssignment);

            // Add the task to the ready set if all dependencies are done
            // Only process child tasks since they are the ones becoming ready
            for (var childId : task.getChildIds()) {
                var childTaskId = new WorkflowTaskId(workflow.getId(), childId);
                pendingDependencies.removeOneDependency(childTaskId);
                if (pendingDependencies.hasNoDependency(childTaskId)) {
                    // This was the last dependency
                    ready.add(childTaskId);
                }
            }
        }

        return new ArrayList<>(vmAssignmentMap.values());
    }

    /// A private record to represent a VM ID.
    private record VmId(int vmId) {
    }

    /// A private record to represent a workflow ID.
    private record WorkflowId(int workflowId) {
    }

    /// A private record to represent a workflow task ID.
    private record WorkflowTaskId(int workflowId, int taskId) {
    }
}
