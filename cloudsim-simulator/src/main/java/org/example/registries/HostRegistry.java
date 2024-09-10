package org.example.registries;

import lombok.Getter;
import org.example.entities.MonitoredHost;
import org.example.tables.HostTable;

import java.util.ArrayList;
import java.util.List;

public class HostRegistry {
    @Getter
    private static final HostRegistry instance = new HostRegistry();

    private final List<MonitoredHost> hosts = new ArrayList<>();

    private HostRegistry() {
    }

    public void updateUtilizationOfHosts(double timeMs) {
        hosts.forEach(host -> host.updateUtilization(timeMs));
    }

    public void registerNewHost(MonitoredHost host) {
        this.hosts.add(host);
    }

    public int getTotalAllocatedVms() {
        return hosts.stream()
                .mapToInt(h -> h.getGuestList().size())
                .sum();
    }

    public double getTotalPowerConsumptionW() {
        return hosts.stream()
                .mapToDouble(MonitoredHost::getAveragePowerConsumption)
                .sum();
    }

    public void printSummaryTable() {
        new HostTable(hosts).print();
    }
}
