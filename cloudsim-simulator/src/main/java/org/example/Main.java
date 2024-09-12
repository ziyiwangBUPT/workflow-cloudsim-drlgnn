package org.example;

import org.cloudbus.cloudsim.Log;
import org.example.api.scheduler.gym.GymSharedQueue;
import org.example.api.scheduler.gym.types.AgentResult;
import org.example.api.scheduler.gym.types.ReleaserAction;
import org.example.api.scheduler.gym.types.ReleaserObservation;
import org.example.api.scheduler.impl.GymWorkflowReleaser;
import org.example.api.scheduler.impl.LocalWorkflowExecutor;
import org.example.api.scheduler.impl.RoundRobinWorkflowScheduler;
import org.example.dataset.Dataset;
import org.example.simulation.SimulatedWorld;
import org.example.simulation.SimulatedWorldConfig;
import org.example.simulation.external.Py4JConnector;

import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws InterruptedException {
        Log.disable();

        // Read input from stdin
        var scanner = new Scanner(System.in);
        var dataset = Dataset.fromJson(scanner.nextLine());

        // Configure simulation
        var config = SimulatedWorldConfig.builder()
                .simulationDuration(1000)
                .schedulingInterval(10)
                .monitoringUpdateInterval(5)
                .build();

        // Create shared queues
        var sharedReleaseQueue = new GymSharedQueue<ReleaserObservation, ReleaserAction>();
        var _ = new GymSharedQueue<ReleaserObservation, ReleaserAction>();

        // Create releaser, scheduler, and executor
        var releaser = new GymWorkflowReleaser(sharedReleaseQueue);
        var scheduler = new RoundRobinWorkflowScheduler();
        var executor = new LocalWorkflowExecutor();

        // Thread for Py4J connector
        var releaserConnector = new Py4JConnector<>(sharedReleaseQueue);
        var releaserThread = new Thread(releaserConnector);
        releaserThread.start();

        // Run simulation
        var world = SimulatedWorld.builder().dataset(dataset)
                .releaser(releaser).scheduler(scheduler).executor(executor)
                .config(config).build();
        var solution = world.runSimulation();
        System.out.println(solution.toJson());

        // Stop Py4J connector
        sharedReleaseQueue.setObservation(AgentResult.truncated());
        releaserThread.interrupt();
        releaserThread.join();
    }
}
