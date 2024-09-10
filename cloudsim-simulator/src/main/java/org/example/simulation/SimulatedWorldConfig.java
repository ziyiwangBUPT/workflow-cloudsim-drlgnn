package org.example.simulation;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class SimulatedWorldConfig {
    private final int simulationDurationSeconds;
    private final int schedulingIntervalSeconds;
    private final int monitoringUpdateIntervalSeconds;
}
