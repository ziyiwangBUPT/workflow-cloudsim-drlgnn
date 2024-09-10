package org.example.factories;

import lombok.Builder;
import org.cloudbus.cloudsim.CloudletSchedulerTimeShared;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Vm;
import org.example.models.DatasetVm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

@Builder
public class VmFactory {
    private Vm createVm(int brokerId, DatasetVm datasetVm) {
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

    public List<Vm> createVms(int brokerId, List<DatasetVm> datasetVms, List<? extends Host> hosts) {
        // Create a map of vm id to mapped host
        var hostMap = new HashMap<Integer, Host>();
        for (var host : hosts) {
            hostMap.put(host.getId(), host);
        }

        var vmList = new ArrayList<Vm>();
        for (var datasetVm : datasetVms) {
            var vm = createVm(brokerId, datasetVm);
            vmList.add(vm);
            vm.setHost(hostMap.get(datasetVm.getHostId()));
        }
        return vmList;
    }
}
