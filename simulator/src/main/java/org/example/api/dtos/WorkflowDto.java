package org.example.api.dtos;

import lombok.Builder;
import lombok.Data;
import org.example.dataset.DatasetWorkflow;

import java.util.List;
import java.util.stream.Collectors;

@Data
@Builder
public class WorkflowDto {
    private final int id;
    private final List<TaskDto> tasks;

    /// Convert from a dataset workflow to a workflow DTO
    public static WorkflowDto from(DatasetWorkflow workflow) {
        return WorkflowDto.builder()
                .id(workflow.getId())
                .tasks(workflow.getTasks().stream().map(
                        task -> TaskDto.builder()
                                .id(task.getId())
                                .workflowId(task.getWorkflowId())
                                .length(task.getLength())
                                .reqMemoryMb(task.getReqMemoryMb())
                                .childIds(task.getChildIds())
                                .build()
                ).collect(Collectors.toList()))
                .build();
    }
}
