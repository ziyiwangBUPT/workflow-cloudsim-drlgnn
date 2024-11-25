package org.example.simulation.listeners;

import lombok.Builder;
import org.example.core.registries.HostRegistry;

/// Tick listener that updates the utilization of monitored hosts.
public class UtilizationUpdater extends SimulationTickListener {
    private static final String NAME = "MONITORED_HOSTS_UPDATER";

    private final long monitoringUpdateInterval;

    private double nextScheduleAtMs = 0;

    @Builder
    protected UtilizationUpdater(long monitoringUpdateInterval) {
        super(NAME);
        this.monitoringUpdateInterval = monitoringUpdateInterval;
    }

    @Override
    protected void onTick(double time) {
        if (nextScheduleAtMs <= time) {
            nextScheduleAtMs += monitoringUpdateInterval;
            var hostRegistry = HostRegistry.getInstance();
            hostRegistry.updateUtilizationOfHosts(time);
        }
    }
}
