package org.example.factories;

import lombok.Builder;
import lombok.NonNull;
import org.cloudbus.cloudsim.*;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/// Factory for creating Datacenter instances.
@Builder
public class DatacenterFactory {
    private static final AtomicInteger CURRENT_DC_ID = new AtomicInteger(0);

    // DC characteristics
    @Builder.Default
    private final String dcArchitecture = "x86";
    @Builder.Default
    private final String dcOperatingSystem = "Linux";
    @Builder.Default
    private final String dcVmm = "Xen";
    @Builder.Default
    private final int dcTimezone = 0;

    // DC cost parameters
    @Builder.Default
    private final double costPerCpu = 0.1;
    @Builder.Default
    private final double costPerMem = 0.1;
    @Builder.Default
    private final double costPerStorage = 0.1;
    @Builder.Default
    private final double costPerBw = 0.1;

    // DC scheduling interval
    @Builder.Default
    private final int schedulingInterval = 1;

    /// Creates a data center with the specified hosts.
    public Datacenter createDatacenter(@NonNull List<? extends Host> hosts) {
        var name = String.format("DC-%d", CURRENT_DC_ID.getAndIncrement());
        var characteristics = new DatacenterCharacteristics(dcArchitecture, dcOperatingSystem, dcVmm, hosts, dcTimezone,
                costPerCpu, costPerMem, costPerStorage, costPerBw);
        var allocationPolicy = new VmAllocationPolicySimple(hosts);
        try {
            return new Datacenter(name, characteristics, allocationPolicy, List.of(), schedulingInterval);
        } catch (Exception e) {
            Log.println("Failed to create data center");
            throw new RuntimeException(e);
        }
    }
}
