package org.example.simulation.external;

import org.example.api.scheduler.gym.GymAgent;
import org.example.api.scheduler.gym.GymSharedQueue;
import org.example.api.scheduler.gym.types.AgentResult;
import py4j.GatewayServer;

import java.util.concurrent.Semaphore;

/// Represents a connector between Python and Java via Py4J.
/// This is used to communicate between the Java and Python sides.
/// This should be run as a separate thread.
public class Py4JConnector<TObservation, TAction> implements Runnable {
    private final GymAgent<TObservation, TAction> agent;
    private final Semaphore shutdownSemaphore = new Semaphore(0);

    public Py4JConnector(GymSharedQueue<TObservation, TAction> queue) {
        this.agent = new GymAgent<>(queue);
    }

    /// Takes a step in the environment.
    public AgentResult<TObservation> step(TAction action) {
        var result = agent.step(action);
        if (result.isTruncated() || result.isTerminated()) {
            shutdownSemaphore.release();
        }

        return result;
    }

    /// Resets the environment.
    public TObservation reset() {
        return agent.reset();
    }

    @Override
    public void run() {
        var server = new GatewayServer(this);
        server.start();

        try {
            // Wait for the shutdown signal
            // After the shutdown signal, wait for a while to allow the server to stop
            shutdownSemaphore.acquire();
            Thread.sleep(1000);
            server.shutdown();
        } catch (InterruptedException ignored) {
        }
    }
}
