package org.example.utils;

import org.cloudbus.cloudsim.power.models.PowerModel;

public class LinearPowerModel implements PowerModel {
    private final double powerIdle;
    private final double powerPeak;

    public LinearPowerModel(double powerIdle, double powerPeak) {
        this.powerIdle = powerIdle;
        this.powerPeak = powerPeak;
    }

    @Override
    public double getPower(double utilization) throws IllegalArgumentException {
        if (utilization < 0 || utilization > 1) {
            throw new IllegalArgumentException("Utilization value must be between 0 and 1");
        }
        return utilization * (powerPeak - powerIdle) + powerIdle;
    }
}
