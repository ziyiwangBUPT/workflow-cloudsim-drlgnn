package org.example.api.scheduler.gym.types;


import lombok.Data;

@Data
public class JsonObservation {
    private final String vmJson;
    private final String taskJson;
}
