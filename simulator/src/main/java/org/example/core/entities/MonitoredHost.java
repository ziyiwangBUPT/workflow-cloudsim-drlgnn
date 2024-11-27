package org.example.core.entities;

import lombok.Builder;
import lombok.Data;
import lombok.Getter;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.VmScheduler;
import org.cloudbus.cloudsim.power.models.PowerModel;
import org.cloudbus.cloudsim.provisioners.BwProvisioner;
import org.cloudbus.cloudsim.provisioners.RamProvisioner;

import java.util.ArrayList;
import java.util.List;

/// Represents a host that can be monitored for power consumption.
public class MonitoredHost extends Host {
    @Getter
    private final PowerModel powerModel;
    private final ArrayList<Utilization> hostUtilizationHistory = new ArrayList<>();

    @Builder
    public MonitoredHost(int id, RamProvisioner ramProvisioner, BwProvisioner bwProvisioner, long storage,
                         List<? extends Pe> peList, VmScheduler vmScheduler, PowerModel powerModel) {
        super(id, ramProvisioner, bwProvisioner, storage, peList, vmScheduler);
        this.powerModel = powerModel;
    }

    /// Update the utilization of the host.
    /// This method should be called periodically to update the host's utilization.
    public void updateUtilization(double time) {
        var totalAllocationMips = 0d;
        for (var vm : getGuestList()) {
            totalAllocationMips += vm.getTotalUtilizationOfCpuMips(time);
        }

        // Assuming delta t = 1
        var utilization = totalAllocationMips / getTotalMips();
        var totalEnergyConsumption = powerModel.getPower(utilization);
        var passiveEnergyConsumption = powerModel.getPower(0);

        var utilizationEntry = Utilization.builder()
                .utilization(utilization)
                .passiveEnergyConsumption(passiveEnergyConsumption)
                .totalEnergyConsumption(totalEnergyConsumption)
                .build();
        this.hostUtilizationHistory.add(utilizationEntry);
    }

    /// Get the average CPU utilization of the host.
    public double getAverageCpuUtilization() {
        return this.hostUtilizationHistory.stream().mapToDouble(Utilization::getUtilization).average()
                .orElse(0);
    }

    /// Get the average power consumption of the host.
    public double getAveragePowerConsumption() {
        return this.hostUtilizationHistory.stream().mapToDouble(Utilization::getTotalEnergyConsumption)
                .average().orElse(0);
    }

    public double getTotalEnergyConsumption() {
        return this.hostUtilizationHistory.stream().mapToDouble(Utilization::getTotalEnergyConsumption)
                .sum();
    }

    public double getActiveEnergyConsumption() {
        return this.hostUtilizationHistory.stream().mapToDouble(Utilization::getActiveEnergyConsumption)
                .sum();
    }

    @Builder
    @Data
    private static class Utilization {
        private double utilization;
        private double totalEnergyConsumption;
        private double passiveEnergyConsumption;

        double getActiveEnergyConsumption() {
            return totalEnergyConsumption - passiveEnergyConsumption;
        }
    }
}
