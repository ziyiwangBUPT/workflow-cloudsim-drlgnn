package org.example.entities;

import lombok.Builder;
import lombok.Getter;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.VmScheduler;
import org.cloudbus.cloudsim.power.models.PowerModel;
import org.cloudbus.cloudsim.provisioners.BwProvisioner;
import org.cloudbus.cloudsim.provisioners.RamProvisioner;

import java.util.List;

/// Represents a host that can be monitored for power consumption.
public class MonitoredHost extends Host {
    @Getter
    private final PowerModel powerModel;
    private double currentAllocatedMips = 0.0;
    private double allocatedMipsRecordSum = 0.0;
    private long allocatedMipsRecordCount = 0;

    @Builder
    public MonitoredHost(int id, RamProvisioner ramProvisioner, BwProvisioner bwProvisioner, long storage,
                         List<? extends Pe> peList, VmScheduler vmScheduler, PowerModel powerModel) {
        super(id, ramProvisioner, bwProvisioner, storage, peList, vmScheduler);
        this.powerModel = powerModel;
    }

    /// Update the utilization of the host.
    /// This method should be called periodically to update the host's utilization.
    public void updateUtilization(double timeMs) {
        var totalAllocationMips = 0d;
        for (var vm : getGuestList()) {
            totalAllocationMips += vm.getTotalUtilizationOfCpuMips(timeMs);
        }

        this.currentAllocatedMips = totalAllocationMips;
        this.allocatedMipsRecordSum += totalAllocationMips;
        this.allocatedMipsRecordCount++;
    }

    /// Get the total MIPS of the host.
    public double getCurrentCpuUtilization() {
        return currentAllocatedMips / getTotalMips();
    }

    /// Get the current power consumption of the host.
    public double getCurrentPowerConsumption() {
        return powerModel.getPower(getCurrentCpuUtilization());
    }

    /// Get the average CPU utilization of the host.
    public double getAverageCpuUtilization() {
        if (allocatedMipsRecordCount == 0) return 0.0;
        return allocatedMipsRecordSum / (allocatedMipsRecordCount * getTotalMips());
    }

    /// Get the average power consumption of the host.
    public double getAveragePowerConsumption() {
        if (getAverageCpuUtilization() == 0) return 0.0;
        return powerModel.getPower(getAverageCpuUtilization());
    }
}
