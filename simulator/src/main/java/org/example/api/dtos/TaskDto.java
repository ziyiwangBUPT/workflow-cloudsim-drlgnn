package org.example.api.dtos;

import lombok.Builder;
import lombok.Data;

import java.util.List;

@Data
@Builder
public class TaskDto {
    private final int id;
    private final int workflowId;
    private final int length;
    private final int reqMemoryMb;
    private final List<Integer> childIds;
}
