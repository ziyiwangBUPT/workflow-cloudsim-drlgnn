package org.example.models;

import lombok.Data;

@Data
public class DatasetHost {
    private final int id;
    private final int cores;
    private final int cpuSpeedMips;
    private final int memoryMb;
    private final int diskMb;
    private final int bandwidthMbps;
    private final int powerIdleWatt;
    private final int powerPeakWatt;
}
