package org.example.api.scheduler.gym.types;


import lombok.Data;

@Data
public class Observation {
    private final String vmJson;
    private final String taskJson;
}
