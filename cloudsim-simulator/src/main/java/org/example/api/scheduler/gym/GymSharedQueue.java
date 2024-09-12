package org.example.api.scheduler.gym;

import org.example.api.scheduler.gym.types.AgentResult;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

/// Represents a shared queue between the Gym environment and the Gym agent.
/// This is used to communicate between the Java and Python sides.
public class GymSharedQueue<TObservation, TAction> {
    private final BlockingQueue<AgentResult<TObservation>> observationQueue = new ArrayBlockingQueue<>(1);
    private final BlockingQueue<TAction> actionQueue = new ArrayBlockingQueue<>(1);

    /// Gets the observation from the queue.
    /// This will block until an observation is available.
    public AgentResult<TObservation> getObservation() {
        try {
            System.err.println("Waiting for observation...");
            var observation = observationQueue.take();
            System.err.println("Got observation: " + observation);
            return observation;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
    }

    /// Sets the observation in the queue.
    public void setObservation(AgentResult<TObservation> observation) {
        try {
            System.err.println("Setting observation: " + observation);
            observationQueue.put(observation);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
    }

    /// Gets the action from the queue.
    /// This will block until an action is available.
    public TAction getAction() {
        try {
            System.err.println("Waiting for action...");
            var action = actionQueue.take();
            System.err.println("Got action: " + action);
            return action;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
    }

    /// Sets the action in the queue.
    /// This will block until the action is available.
    public void setAction(TAction action) {
        try {
            System.err.println("Setting action: " + action);
            actionQueue.put(action);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
    }
}
