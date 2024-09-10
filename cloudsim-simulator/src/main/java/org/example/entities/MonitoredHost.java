package org.example.entities;

import lombok.Builder;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.VmScheduler;
import org.cloudbus.cloudsim.power.models.PowerModel;
import org.cloudbus.cloudsim.provisioners.BwProvisioner;
import org.cloudbus.cloudsim.provisioners.RamProvisioner;

import java.util.List;

public class MonitoredHost extends Host {
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

    public void updateUtilization(double timeMs) {
        var totalAllocationMips = 0d;
        for (var vm : getGuestList()) {
            totalAllocationMips += vm.getTotalUtilizationOfCpuMips(timeMs);
        }

        this.currentAllocatedMips = totalAllocationMips;
        this.allocatedMipsRecordSum += totalAllocationMips;
        this.allocatedMipsRecordCount++;
    }

    public double getCurrentCpuUtilization() {
        return currentAllocatedMips / getTotalMips();
    }

    public double getCurrentPowerConsumption() {
        return powerModel.getPower(getCurrentCpuUtilization());
    }

    public double getAverageCpuUtilization() {
        if (allocatedMipsRecordCount == 0) return 0.0;
        return allocatedMipsRecordSum / (allocatedMipsRecordCount * getTotalMips());
    }

    public double getAveragePowerConsumption() {
        if (getAverageCpuUtilization() == 0) return 0.0;
        return powerModel.getPower(getAverageCpuUtilization());
    }
}
