package org.example.core.entities;

import org.cloudbus.cloudsim.VmAllocationPolicy;
import org.cloudbus.cloudsim.core.GuestEntity;
import org.cloudbus.cloudsim.core.HostEntity;
import org.example.dataset.DatasetVm;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/// Represents a VM allocation policy that assigns VMs to hosts based on the dataset.
public class DatasetVmAllocationPolicy extends VmAllocationPolicy {
    private final Map<Integer, HostEntity> assignedHosts;

    public DatasetVmAllocationPolicy(List<? extends HostEntity> hosts, List<DatasetVm> vmDataset) {
        super(hosts);

        this.assignedHosts = new HashMap<>();
        var hostMap = new HashMap<Integer, HostEntity>();
        hosts.forEach(host -> hostMap.put(host.getId(), host));
        vmDataset.forEach(vm -> assignedHosts.put(vm.getId(), hostMap.get(vm.getHostId())));
    }

    @Override
    public HostEntity findHostForGuest(GuestEntity guestEntity) {
        return assignedHosts.get(guestEntity.getId());
    }
}
