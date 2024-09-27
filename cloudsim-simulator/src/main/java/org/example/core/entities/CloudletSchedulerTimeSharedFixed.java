package org.example.core.entities;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.CloudletSchedulerTimeShared;

import java.util.List;

/// A cloudlet scheduler that fixes some bugs in cloudsim
public class CloudletSchedulerTimeSharedFixed extends CloudletSchedulerTimeShared {
    @Override
    public double getTotalCurrentAvailableMipsForCloudlet(Cloudlet cl, List<Double> mipsShare) {
        // The original implementation multiplies this with the number of PEs
        // However, in several places (eg: cloudletSubmit) the estimated finish time is calculated
        // using the current capacity only
        // Semantically, if we consider the MI are for one core in the cloudlet (maximum)
        // then the available mips for cloudlet one parallel process will be the current capacity
        return getCurrentCapacity();
    }

    @Override
    public double getEstimatedFinishTime(Cloudlet cl, double time) {
        // https://github.com/Cloudslab/cloudsim/issues/77
        // estimated finish time should not include the current time since this is used to schedule
        // Otherwise time will be added twice
        return super.getEstimatedFinishTime(cl, time) - time;
    }
}
