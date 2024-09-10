package org.example.models;

import lombok.Data;

import java.util.List;

@Data
public class Dataset {
    private final List<DatasetWorkflow> workflows;
    private final List<DatasetVm> vms;
    private final List<DatasetHost> hosts;
}
