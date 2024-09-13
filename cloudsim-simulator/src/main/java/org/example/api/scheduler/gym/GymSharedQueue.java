package org.example.api.scheduler.gym;

import org.example.api.scheduler.gym.types.AgentResult;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

/// Represents a shared queue between the Gym environment and the Gym agent.
/// This is used to communicate between the Java and Python sides.
public class GymSharedQueue<TObservation, TAction> {
    private static final int MAX_WAIT_FOR_OBS = 1000; // Seconds
    private static final int MAX_WAIT_FOR_ACTION = 10; // Seconds

    private final BlockingQueue<AgentResult<TObservation>> observationQueue = new ArrayBlockingQueue<>(1);
    private final BlockingQueue<TAction> actionQueue = new ArrayBlockingQueue<>(1);

    /// Gets the observation from the queue.
    /// This will block until an observation is available.
    public AgentResult<TObservation> getObservation() {
        try {
            var result = observationQueue.poll(MAX_WAIT_FOR_OBS, TimeUnit.SECONDS);
            if (result == null) throw new InterruptedException("Timeout on getting observation");
            return result;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
    }

    /// Sets the observation in the queue.
    public void setObservation(AgentResult<TObservation> observation) {
        try {
            observationQueue.put(observation);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
    }

    /// Gets the action from the queue.
    /// This will block until an action is available.
    /// If there is no action for some time, this will return
    public TAction getAction() {
        try {
            var result = actionQueue.poll(MAX_WAIT_FOR_ACTION, TimeUnit.SECONDS);
            if (result == null) throw new InterruptedException("Timeout on getting action");
            return result;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
    }

    /// Sets the action in the queue.
    /// This will block until the action is available.
    public void setAction(TAction action) {
        try {
            actionQueue.put(action);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
    }
}
