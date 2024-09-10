package org.example.models;

import lombok.Data;

import java.util.List;

@Data
public class DatasetWorkflow {
    private final int id;
    private final List<DatasetTask> tasks;
    private final int arrivalTime;
}
