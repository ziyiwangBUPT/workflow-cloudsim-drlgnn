package org.example;

import lombok.Setter;
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
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.io.File;
import java.util.concurrent.Callable;

@Setter
@Command(name = "CloudSim Simulator", mixinStandardHelpOptions = true, version = "1.0",
        description = "Runs a simulation of a workflow scheduling algorithm.")
public class Application implements Callable<Integer> {
    @Option(names = {"-f", "--file"}, description = "Dataset file")
    private File datasetFile;

    @Option(names = {"-d", "--duration"}, description = "Duration of the simulation")
    private int simulationDuration = 1000;

    @Override
    public Integer call() throws Exception {
        System.err.println("Running simulation...");
        Log.disable();

        // Read input file or stdin
        var dataset = datasetFile != null
                ? Dataset.fromFile(datasetFile)
                : Dataset.fromStdin();

        // Configure simulation
        var config = SimulatedWorldConfig.builder()
                .simulationDuration(simulationDuration)
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
        releaserThread.join();
        return 0;
    }

    public static void main(String[] args) {
        int exitCode = new CommandLine(new Application()).execute(args);
        System.exit(exitCode);
    }
}
