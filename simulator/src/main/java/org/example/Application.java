package org.example;

import lombok.Setter;
import org.cloudbus.cloudsim.Log;
import org.example.api.scheduler.gym.GymSharedQueue;
import org.example.api.scheduler.gym.WorkflowSchedulerFactory;
import org.example.api.scheduler.gym.types.AgentResult;
import org.example.api.scheduler.gym.types.StaticAction;
import org.example.api.scheduler.gym.types.StaticObservation;
import org.example.api.executor.LocalWorkflowExecutor;
import org.example.core.registries.CloudletRegistry;
import org.example.core.registries.HostRegistry;
import org.example.dataset.Dataset;
import org.example.sensors.RewardSensor;
import org.example.sensors.TaskStateSensor;
import org.example.simulation.SimulatedWorld;
import org.example.simulation.SimulatedWorldConfig;
import org.example.simulation.external.Py4JConnector;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.io.File;
import java.util.concurrent.Callable;

@Setter
@Command(name = "cloudsim-simulator", mixinStandardHelpOptions = true, version = "1.0",
        description = "Runs a simulation of a workflow scheduling algorithm.",
        usageHelpAutoWidth = true)
public class Application implements Callable<Integer> {
    @Option(names = {"-f", "--file"}, description = "Dataset file")
    private File datasetFile;

    @Option(names = {"-p", "--port"}, description = "Py4J port", defaultValue = "25333")
    private int py4JPort;

    @Option(names = {"-a", "--algorithm"}, description = "Scheduling algorithm", defaultValue = "static:round-robin")
    private String algorithm;

    @Override
    public Integer call() throws Exception {
        System.err.println("Running simulation...");
        Log.disable();

        // Read input file or stdin
        var dataset = datasetFile != null
                ? Dataset.fromFile(datasetFile)
                : Dataset.fromStdin();
        var duration = dataset.maximumHorizon();

        // Configure simulation
        var config = SimulatedWorldConfig.builder()
                .simulationDuration(duration)
                .monitoringUpdateInterval(1)
                .build();

        // Create shared queue
        var gymSharedQueue = new GymSharedQueue<StaticObservation, StaticAction>();

        // Create scheduler, and executor
        var scheduler = new WorkflowSchedulerFactory().staticSharedQueue(gymSharedQueue)
                .create(algorithm);
        var executor = new LocalWorkflowExecutor();

        // Thread for Py4J connector
        var gymConnector = new Py4JConnector<>(py4JPort, gymSharedQueue);
        var gymThread = new Thread(gymConnector);
        gymThread.start();

        // Run simulation
        var world = SimulatedWorld.builder().dataset(dataset)
                .scheduler(scheduler).executor(executor)
                .config(config).build();
        var solution = world.runSimulation();

        var rewardSensor = RewardSensor.getInstance();
        var hostRegistry = HostRegistry.getInstance();
        var reward = rewardSensor.finalReward(duration);

        AgentResult<StaticObservation> finalAgentResult = AgentResult.truncated(reward);
        finalAgentResult.addInfo("solution", solution.toJson());
        finalAgentResult.addInfo("total_energy_consumption_j", Double.toString(hostRegistry.getTotalEnergyConsumptionJ()));
        finalAgentResult.addInfo("active_energy_consumption_j", Double.toString(hostRegistry.getActiveEnergyConsumptionJ()));
        gymSharedQueue.setObservation(finalAgentResult);

        var cloudletRegistry = CloudletRegistry.getInstance();
        var taskStateSensor = TaskStateSensor.getInstance();
        cloudletRegistry.printSummaryTable();
        System.err.printf("Total makespan (s)           : %.5f%n", cloudletRegistry.getMakespan());
        System.err.printf("Total power consumption (W)  : %.2f%n", hostRegistry.getTotalPowerConsumptionW());
        System.err.printf("Total allocated VMs          : %d / %d%n", hostRegistry.getTotalAllocatedVms(), world.getBroker().getGuestList().size());
        System.err.printf("Unfinished Cloudlets         : %d / %d%n", cloudletRegistry.getRunningCloudletCount(), cloudletRegistry.getSize());
        System.err.printf("Total Cloudlet length (MI)   : %d%n", cloudletRegistry.getTotalCloudletLength());
        System.err.printf("Last task finish time (s)    : %.2f%n", cloudletRegistry.getLastCloudletFinishedAt());
        System.err.printf("Buffered Tasks               : %d%n", taskStateSensor.getBufferedTasks());
        System.err.printf("Scheduled Tasks              : %d%n", taskStateSensor.getScheduledTasks());
        System.err.printf("Executed Tasks               : %d%n", taskStateSensor.getExecutedTasks());
        System.err.printf("Finished Tasks               : %d%n", taskStateSensor.getCompletedTasks());
        System.out.println(solution.toJson());

        // Stop Py4J connector
        gymThread.join(5000);
        return 0;
    }

    public static void main(String[] args) {
        int exitCode = new CommandLine(new Application()).execute(args);
        System.exit(exitCode);
    }
}
