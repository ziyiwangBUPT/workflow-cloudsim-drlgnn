package org.example.api.scheduler.rl;

import lombok.NonNull;
import org.example.api.scheduler.WorkflowReleaser;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;

import java.util.Map;

public abstract class RlWorkflowReleaser implements WorkflowReleaser {
    @Override
    public void notifyNewVm(@NonNull VmDto newVm) {
    }

    @Override
    public void notifyNewWorkflow(@NonNull WorkflowDto newWorkflow) {
    }

    @Override
    public synchronized boolean shouldRelease() {
        return false;
    }

    protected synchronized Result step(Action action) {
        return null;
    }

    protected record Action(boolean release) {
    }

    protected record Observation() {
    }

    protected record Result(
            Observation observation, // Next observation due to the agent actions
            double reward, // The reward as a result of taking the action
            boolean terminated, // Whether the agent reaches the terminal state
            boolean truncated, // Whether the truncation outside the scope of the MDP is satisfied (time limit)
            Map<String, Observation> info // Contains auxiliary diagnostic information
    ) {
    }
}
