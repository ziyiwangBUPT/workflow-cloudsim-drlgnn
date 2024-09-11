package org.example.api;

import lombok.NonNull;
import org.example.api.dtos.TaskAssignmentDto;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.WorkflowDto;

import java.util.List;

/// The interface for the workflow executor.
public interface WorkflowExecutor {
    /// Submits the assignments to the executor.
    /// This would merge the new assignments to the current execution graph of the workflows.
    void submitAssignments(@NonNull List<WorkflowDto> workflows, @NonNull List<VmAssignmentDto> assignments);

    /// Notify the completion of a task.
    /// Will compute the ready queue based on the completed task.
    void notifyCompletion(int workflowId, int taskId);

    /// Polls the task assignment instructions.
    /// Once this is called, the ready queue is emptied.
    List<TaskAssignmentDto> pollTaskAssignments();
}
