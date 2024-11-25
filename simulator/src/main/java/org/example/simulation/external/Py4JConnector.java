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
    private static final int SHUTDOWN_COUNTDOWN = 1000;

    private final GymAgent<TObservation, TAction> agent;
    private final Semaphore shutdownSemaphore = new Semaphore(0);

    private final int port;

    public Py4JConnector(int port, GymSharedQueue<TObservation, TAction> queue) {
        this.port = port;
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
        var server = new GatewayServer(this, port);
        server.start();

        try {
            // Wait for the shutdown signal
            // After the shutdown signal, wait for a while to allow the server to stop
            shutdownSemaphore.acquire();
            Thread.sleep(SHUTDOWN_COUNTDOWN);
            server.shutdown();
        } catch (InterruptedException ignored) {
        }
    }
}
