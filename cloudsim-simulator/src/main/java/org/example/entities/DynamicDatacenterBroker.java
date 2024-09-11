package org.example.entities;

import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.DatacenterBroker;
import org.cloudbus.cloudsim.core.SimEvent;

/// Datacenter that can schedule cloudlets dynamically.
public class DynamicDatacenterBroker extends DatacenterBroker {
    public DynamicDatacenterBroker(String name) throws Exception {
        super(name);
    }

    /// Schedules all submitted cloudlets. (Dynamic scheduling)
    /// Note that this will throw an exception if any cloudlet has not been scheduled.
    public void scheduleSubmittedCloudlets() {
        // No cloudlet should have -1 as the guest (unscheduled)
        for (var cloudlet : getCloudletList()) {
            if (cloudlet.getGuestId() == -1) {
                throw new IllegalStateException("Cloudlet %d was not scheduled".formatted(cloudlet.getCloudletId()));
            }
        }

        submitCloudlets();
    }

    @Override
    protected void processCloudletReturn(SimEvent ev) {
        var cloudlet = (Cloudlet) ev.getData();

        // Modified logic to not exit if all cloudlets have been processed
        getCloudletReceivedList().add(cloudlet);
        cloudletsSubmitted--;
    }
}
