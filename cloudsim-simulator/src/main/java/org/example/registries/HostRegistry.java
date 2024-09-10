package org.example.registries;

import lombok.Getter;
import lombok.NonNull;
import org.example.entities.MonitoredHost;
import org.example.tables.HostTable;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/// A registry of all hosts in the simulation.
public class HostRegistry {
    @Getter
    private static final HostRegistry instance = new HostRegistry();

    private final Map<Integer, MonitoredHost> hosts = new HashMap<>();

    private HostRegistry() {
    }

    /// Update the utilization of all hosts in the registry.
    public void updateUtilizationOfHosts(double timeMs) {
        hosts.values().forEach(host -> host.updateUtilization(timeMs));
    }

    /// Register a new list of hosts.
    public void registerNewHosts(@NonNull List<MonitoredHost> newHosts) {
        newHosts.forEach(host -> hosts.put(host.getId(), host));
    }

    /// Get a host by its ID.
    public MonitoredHost getHost(int id) {
        return hosts.get(id);
    }

    /// Get the total number of allocated VMs.
    public int getTotalAllocatedVms() {
        return hosts.values().stream().mapToInt(h -> h.getGuestList().size()).sum();
    }

    /// Get the total power consumption of all hosts.
    public double getTotalPowerConsumptionW() {
        return hosts.values().stream().mapToDouble(MonitoredHost::getAveragePowerConsumption).sum();
    }

    /// Print a summary table of all hosts.
    public void printSummaryTable() {
        new HostTable(hosts.values()).print();
    }
}
