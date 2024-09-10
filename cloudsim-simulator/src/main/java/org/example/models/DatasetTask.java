package org.example.models;

import lombok.Data;

import java.util.List;

@Data
public class DatasetTask {
    private final int id;
    private final int length;
    private final int reqCores;
    private final List<Integer> childIds;
}
