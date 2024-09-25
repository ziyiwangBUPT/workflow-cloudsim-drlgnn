package org.example.api.dtos;

import lombok.Builder;
import lombok.Data;
import lombok.NonNull;
import org.cloudbus.cloudsim.Vm;
import org.example.core.entities.MonitoredHost;

@Data
@Builder
public class VmDto {
    private final int id;
    private final HostDto host;
    private final int cores;
    private final double cpuSpeedMips;
    private final int memoryMb;
    private final long diskMb;
    private final long bandwidthMbps;
    private final String vmm;

    /// Convert a Vm entity to a DTO
    public static VmDto from(@NonNull Vm vm) {
        var host = (MonitoredHost) vm.getHost();
        return VmDto.builder()
                .id(vm.getId())
                .host(HostDto.builder()
                        .id(host.getId())
                        .cores(host.getNumberOfPes())
                        .cpuSpeedMips(host.getTotalMips())
                        .memoryMb(host.getRam())
                        .diskMb(host.getStorage())
                        .bandwidthMbps(host.getBw())
                        .powerIdleWatt(host.getPowerModel().getPower(0))
                        .powerPeakWatt(host.getPowerModel().getPower(1))
                        .build())
                .cores(vm.getNumberOfPes())
                .cpuSpeedMips(vm.getMips())
                .memoryMb(vm.getRam())
                .diskMb(vm.getSize())
                .bandwidthMbps(vm.getBw())
                .vmm(vm.getVmm())
                .build();
    }

    public boolean canRunTask(TaskDto task) {
        return task.getReqCores() <= cores;
    }
}
