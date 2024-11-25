package org.example.api.scheduler.gym;

import org.example.api.scheduler.gym.types.AgentResult;

/// Represents an agent that interacts with a Gym environment.
/// This is called from the Python side via Py4J.
public class GymAgent<TObservation, TAction> {
    private final GymSharedQueue<TObservation, TAction> queue;

    public GymAgent(GymSharedQueue<TObservation, TAction> queue) {
        this.queue = queue;
    }

    /// Takes a step in the environment.
    /// This will set the action and block, waiting for the environment to return an observation.
    public AgentResult<TObservation> step(TAction action) {
        queue.setAction(action);
        return queue.getObservation();
    }

    /// Resets the environment.
    /// Must be called as the first method, before step.
    public TObservation reset() {
        var observation = queue.getObservation();
        return observation.getObservation();
    }
}
