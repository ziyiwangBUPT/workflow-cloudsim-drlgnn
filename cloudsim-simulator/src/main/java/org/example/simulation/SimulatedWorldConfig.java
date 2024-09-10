package org.example.simulation;

import lombok.Builder;
import lombok.Getter;

/// Configuration for the simulated world.
@Getter
@Builder
public class SimulatedWorldConfig {
    /// How much time the simulation should run for (in seconds).
    private final int simulationDuration;
    /// How often the scheduler should run (in seconds).
    /// This is the time between each (static) scheduling decision.
    private final int schedulingInterval;
    /// How often the monitoring should run (in seconds).
    /// This is the time between each monitoring (eg: power consumption) update.
    private final int monitoringUpdateInterval;
}
