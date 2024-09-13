package org.example.simulation.external;

import org.example.api.scheduler.gym.GymAgent;
import org.example.api.scheduler.gym.GymSharedQueue;
import org.example.api.scheduler.gym.types.AgentResult;
import py4j.GatewayServer;

/// Represents a connector between Python and Java via Py4J.
/// This is used to communicate between the Java and Python sides.
/// This should be run as a separate thread.
public class Py4JConnector<TObservation, TAction> implements Runnable {
    private final GymAgent<TObservation, TAction> agent;

    public Py4JConnector(GymSharedQueue<TObservation, TAction> queue) {
        this.agent = new GymAgent<>(queue);
    }

    /// Takes a step in the environment.
    public AgentResult<TObservation> step(TAction action) {
        return agent.step(action);
    }

    /// Resets the environment.
    public TObservation reset() {
        return agent.reset();
    }

    @Override
    public void run() {
        var server = new GatewayServer(this);
        server.start();
    }
}
