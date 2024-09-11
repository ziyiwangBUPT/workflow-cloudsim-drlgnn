package org.example.api.impl;

import lombok.NonNull;
import org.cloudbus.cloudsim.core.CloudSim;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;
import org.example.api.WorkflowReleaser;

import java.util.List;

/// A simple workflow releaser that releases workflows every T seconds.
public class PeriodicWorkflowReleaser implements WorkflowReleaser {
    private static final int RELEASE_INTERVAL = 25;

    private double lastRelease = 0;
    private boolean hasNewWorkflows = false;

    @Override
    public void submitVms(@NonNull List<VmDto> newVms) {
        // Do nothing
    }

    @Override
    public void submitWorkflows(@NonNull List<WorkflowDto> newWorkflows) {
        hasNewWorkflows = true;
    }

    @Override
    public boolean shouldRelease() {
        // No point in releasing workflows if there are no new workflows
        if (!hasNewWorkflows) return false;

        var currentTime = CloudSim.clock();
        if (currentTime - lastRelease >= RELEASE_INTERVAL) {
            lastRelease = currentTime;
            hasNewWorkflows = false;
            return true;
        }
        return false;
    }
}
