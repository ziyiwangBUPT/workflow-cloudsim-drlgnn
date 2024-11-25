package org.example.api.scheduler.gym;

import org.example.api.scheduler.gym.types.AgentResult;

/// Represents an environment that interacts with a Gym agent.
/// This is called from the Java side.
public class GymEnvironment<TObservation, TAction> {
    private final GymSharedQueue<TObservation, TAction> queue;

    public GymEnvironment(GymSharedQueue<TObservation, TAction> queue) {
        this.queue = queue;
    }

    /// Takes a step in the environment.
    /// This will set the observation and block, waiting for the agent to return an action.
    public TAction step(AgentResult<TObservation> observation) {
        queue.setObservation(observation);
        return queue.getAction();
    }
}
