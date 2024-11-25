package org.example.core.registries;

import lombok.Getter;
import lombok.NonNull;
import org.cloudbus.cloudsim.Vm;
import org.example.utils.SummaryTable;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/// A registry of all VMs in the simulation.
public class VmRegistry extends AbstractRegistry<Vm> {
    @Getter
    private static final VmRegistry instance = new VmRegistry();

    private final Map<Integer, Vm> vmMap;

    private VmRegistry() {
        this.vmMap = new HashMap<>();
    }

    /// Register a new list of vms.
    public void registerNewVms(@NonNull List<Vm> newVms) {
        newVms.forEach(vm -> register(vm.getId(), vm));
        newVms.forEach(vm -> vmMap.put(vm.getId(), vm));
    }

    public Vm getVm(int id) {
        return vmMap.get(id);
    }

    public double estimateMakespan(int vmId, long length) {
        var vm = getVm(vmId);
        return length / vm.getMips();
    }

    @Override
    protected SummaryTable<Vm> buildSummaryTable() {
        var vmSummaryTable = new SummaryTable<Vm>();
        vmSummaryTable.addColumn("VM", SummaryTable.ID_UNIT, SummaryTable.INTEGER_FORMAT, Vm::getId);
        vmSummaryTable.addColumn("Host", SummaryTable.ID_UNIT, SummaryTable.INTEGER_FORMAT, vm -> vm.getHost().getId());
        vmSummaryTable.addColumn("Cores", SummaryTable.COUNT_UNIT, SummaryTable.INTEGER_FORMAT, Vm::getNumberOfPes);
        vmSummaryTable.addColumn("Speed", SummaryTable.GIPS_UNIT, SummaryTable.DECIMAL_FORMAT, Vm::getMips);
        return vmSummaryTable;
    }
}
