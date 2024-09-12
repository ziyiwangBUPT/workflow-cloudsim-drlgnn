package org.example.api.scheduler;

import lombok.NonNull;
import org.example.api.dtos.TaskAssignmentDto;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;

import java.util.List;

/// The interface for the workflow executor.
public interface WorkflowExecutor {
    /// Submit a new VM to the system.
    /// This is called from the coordinator when it discovers a new VM.
    void notifyNewVm(@NonNull VmDto newVm);

    /// Submit a new workflow to the system.
    /// This is called from the coordinator when a workflow gets released.
    void notifyNewWorkflow(@NonNull WorkflowDto newWorkflow);

    /// Submits the assignments to the executor.
    /// This would merge the new assignments to the current execution graph of the workflows.
    void notifyScheduling(@NonNull VmAssignmentDto assignment);

    /// Notify the completion of a task.
    /// Will compute the ready queue based on the completed task.
    void notifyCompletion(int workflowId, int taskId);

    /// Polls the task assignment instructions.
    /// Once this is called, the ready queue is emptied.
    List<TaskAssignmentDto> pollTaskAssignments();
}
