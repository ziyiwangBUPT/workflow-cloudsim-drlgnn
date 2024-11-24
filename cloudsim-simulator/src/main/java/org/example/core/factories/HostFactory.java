package org.example.core.factories;

import lombok.Builder;
import lombok.NonNull;
import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.power.models.PowerModelLinear;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import org.example.dataset.DatasetHost;
import org.example.core.entities.MonitoredHost;
import org.example.core.registries.HostRegistry;

import java.util.ArrayList;
import java.util.List;

/// Factory for creating hosts.
@Builder
public class HostFactory {
    private MonitoredHost createHost(@NonNull DatasetHost datasetHost) {
        // Create PEs using the same MIPS
        var peList = new ArrayList<Pe>();
        var peSpeed = datasetHost.getCpuSpeedMips() / datasetHost.getCores(); // MIPS
        for (int i = 0; i < datasetHost.getCores(); i++) {
            peList.add(new Pe(i, new PeProvisionerSimple(peSpeed)));
        }

        // It is possible to use exact SPEC power models
        // Example: Following a similar implementation to PowerModelSpecPowerHpProLiantMl110G3PentiumD930
        // using https://www.spec.org/power_ssj2008/results/res2019q2/power_ssj2008-20190409-00952.html for R740.
        var maxPower = datasetHost.getPowerPeakWatt();
        var staticPowerPercent = (float) datasetHost.getPowerIdleWatt() / maxPower;
        var powerModel = new PowerModelLinear(maxPower, staticPowerPercent);

        return MonitoredHost.builder()
                .id(datasetHost.getId())
                .ramProvisioner(new RamProvisionerSimple(datasetHost.getMemoryMb()))
                .bwProvisioner(new BwProvisionerSimple(datasetHost.getBandwidthMbps()))
                .storage(datasetHost.getDiskMb())
                .peList(peList)
                .vmScheduler(new VmSchedulerTimeShared(peList))
                .powerModel(powerModel)
                .build();
    }

    /// Create a list of hosts based on the dataset.
    public List<MonitoredHost> createHosts(@NonNull List<DatasetHost> datasetHosts) {
        var hostList = new ArrayList<MonitoredHost>();
        for (var datasetHost : datasetHosts) {
            var host = createHost(datasetHost);
            hostList.add(host);
        }
        var hostRegistry = HostRegistry.getInstance();
        hostRegistry.registerNewHosts(hostList);
        return hostList;
    }
}
