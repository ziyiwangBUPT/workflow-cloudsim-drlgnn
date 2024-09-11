package org.example.api.dtos;

import lombok.Builder;
import lombok.Data;

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
}
