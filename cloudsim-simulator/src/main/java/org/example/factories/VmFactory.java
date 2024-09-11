package org.example.factories;

import lombok.Builder;
import lombok.NonNull;
import org.cloudbus.cloudsim.CloudletSchedulerTimeShared;
import org.cloudbus.cloudsim.Vm;
import org.example.dataset.DatasetVm;
import org.example.registries.HostRegistry;

import java.util.ArrayList;
import java.util.List;

/// Factory for creating VMs.
@Builder
public class VmFactory {
    private Vm createVm(int brokerId, @NonNull DatasetVm datasetVm) {
        var id = datasetVm.getId();
        var vmSpeed = datasetVm.getCpuSpeedMips(); // MIPS
        var vmCores = datasetVm.getCores(); // vCPUs
        var vmRamMb = datasetVm.getMemoryMb(); // MB
        var vmBwMbps = datasetVm.getBandwidthMbps(); // Mbit/s
        var vmSizeMb = datasetVm.getDiskMb(); // MB
        var vmVmm = datasetVm.getVmm();

        var cloudletScheduler = new CloudletSchedulerTimeShared();
        return new Vm(id, brokerId, vmSpeed, vmCores,
                vmRamMb, vmBwMbps, vmSizeMb, vmVmm, cloudletScheduler);
    }

    /// Create a list of VMs based on the dataset using registered hosts.
    public List<Vm> createVms(int brokerId, @NonNull List<DatasetVm> datasetVms) {
        var hostRegistry = HostRegistry.getInstance();

        var vmList = new ArrayList<Vm>();
        for (var datasetVm : datasetVms) {
            var vm = createVm(brokerId, datasetVm);
            var host = hostRegistry.findRegistered(datasetVm.getHostId());
            vm.setHost(host);
            vmList.add(vm);
        }
        return vmList;
    }
}
