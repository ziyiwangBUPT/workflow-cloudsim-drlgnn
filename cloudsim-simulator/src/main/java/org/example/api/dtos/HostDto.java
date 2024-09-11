package org.example.api.dtos;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class HostDto {
    private final int id;
    private final int cores;
    private final double cpuSpeedMips;
    private final int memoryMb;
    private final long diskMb;
    private final long bandwidthMbps;
    private final double powerIdleWatt;
    private final double powerPeakWatt;
}
