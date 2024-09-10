package org.example.ticks;

import lombok.Builder;
import org.example.registries.HostRegistry;

public class MonitoredHostsUpdater extends SimulationTickListener {
    private static final String NAME = "MONITORED_HOSTS_UPDATER";

    private final long monitoringUpdateInterval;

    private double nextScheduleAtMs = 0;

    @Builder
    protected MonitoredHostsUpdater(long monitoringUpdateInterval) {
        super(NAME);
        this.monitoringUpdateInterval = monitoringUpdateInterval;
    }

    @Override
    protected void onTick(double timeMs) {
        if (nextScheduleAtMs <= timeMs) {
            nextScheduleAtMs += monitoringUpdateInterval;
            var hostRegistry = HostRegistry.getInstance();
            hostRegistry.updateUtilizationOfHosts(timeMs);
        }
    }
}
