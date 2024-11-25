package org.example.core.entities;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.CloudletSchedulerTimeShared;

/// A cloudlet scheduler that fixes some bugs in cloudsim
public class CloudletSchedulerTimeSharedFixed extends CloudletSchedulerTimeShared {
    @Override
    public double getEstimatedFinishTime(Cloudlet cl, double time) {
        // https://github.com/Cloudslab/cloudsim/issues/77
        // estimated finish time should not include the current time since this is used to schedule
        // Otherwise time will be added twice
        return super.getEstimatedFinishTime(cl, time) - time;
    }
}
