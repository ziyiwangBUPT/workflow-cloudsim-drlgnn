package org.example.core.registries;

import lombok.Getter;
import lombok.NonNull;
import org.cloudbus.cloudsim.Host;
import org.example.core.entities.MonitoredHost;
import org.example.utils.SummaryTable;

import java.util.List;

/// A registry of all hosts in the simulation.
public class HostRegistry extends AbstractRegistry<MonitoredHost> {
    private static final int K = 1024;

    @Getter
    private static final HostRegistry instance = new HostRegistry();

    private HostRegistry() {
    }

    /// Update the utilization of all hosts in the registry.
    public void updateUtilizationOfHosts(double time) {
        itemStream().forEach(host -> host.updateUtilization(time));
    }

    /// Register a new list of hosts.
    public void registerNewHosts(@NonNull List<MonitoredHost> newHosts) {
        newHosts.forEach(host -> register(host.getId(), host));
    }

    /// Get the total number of allocated VMs.
    public int getTotalAllocatedVms() {
        return itemStream().mapToInt(h -> h.getGuestList().size()).sum();
    }

    /// Get the total power consumption of all hosts.
    public double getTotalPowerConsumptionW() {
        return itemStream().mapToDouble(MonitoredHost::getAveragePowerConsumption).sum();
    }

    /// Get the total energy consumption of all hosts.
    public double getTotalEnergyConsumptionJ() {
        return itemStream().mapToDouble(MonitoredHost::getTotalEnergyConsumption).sum();
    }

    /// Get the total energy consumption of all hosts.
    public double getActiveEnergyConsumptionJ() {
        return itemStream().mapToDouble(MonitoredHost::getActiveEnergyConsumption).sum();
    }

    @Override
    protected SummaryTable<MonitoredHost> buildSummaryTable() {
        var summaryTable = new SummaryTable<MonitoredHost>();

        summaryTable.addColumn("DC", SummaryTable.ID_UNIT, SummaryTable.STRING_FORMAT, host -> host.getDatacenter().getId());
        summaryTable.addColumn("Host", SummaryTable.ID_UNIT, SummaryTable.STRING_FORMAT, Host::getId);
        summaryTable.addColumn("PES", SummaryTable.COUNT_UNIT, SummaryTable.INTEGER_FORMAT, Host::getNumberOfPes);
        summaryTable.addColumn("Speed ", SummaryTable.GIPS_UNIT, SummaryTable.DECIMAL_FORMAT, host -> host.getTotalMips() / K);
        summaryTable.addColumn("Ram", SummaryTable.GB_UNIT, SummaryTable.INTEGER_FORMAT, host -> host.getRam() / K);
        summaryTable.addColumn(" BW  ", SummaryTable.GB_S_UNIT, SummaryTable.DECIMAL_FORMAT, host -> (double) host.getBw() / K);
        summaryTable.addColumn("Storage", SummaryTable.GB_UNIT, SummaryTable.INTEGER_FORMAT, host -> host.getStorage() / 1000);
        summaryTable.addColumn("VMs", SummaryTable.COUNT_UNIT, SummaryTable.INTEGER_FORMAT, host -> host.getGuestList().size());
        summaryTable.addColumn("CPU Usage", SummaryTable.PERC_UNIT, SummaryTable.DECIMAL_FORMAT, host -> host.getAverageCpuUtilization() * 100);

        return summaryTable;
    }
}
