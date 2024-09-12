package org.example.api.scheduler.impl;

import lombok.NonNull;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;
import org.example.api.scheduler.WorkflowReleaser;

/// A simple workflow releaser that releases workflows as soon as they are added.
public class InstantWorkflowReleaser implements WorkflowReleaser {
    private boolean hasNewWorkflows = false;

    @Override
    public void notifyNewVm(@NonNull VmDto newVm) {
        // Do nothing
    }

    @Override
    public void notifyNewWorkflow(@NonNull WorkflowDto newWorkflow) {
        hasNewWorkflows = true;
    }

    @Override
    public boolean shouldRelease() {
        return hasNewWorkflows;
    }
}
