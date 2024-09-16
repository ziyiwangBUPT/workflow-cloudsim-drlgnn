package org.example.simulation;

import lombok.Builder;
import lombok.NonNull;
import org.cloudbus.cloudsim.core.CloudSim;
import org.example.api.scheduler.WorkflowExecutor;
import org.example.api.scheduler.WorkflowReleaser;
import org.example.api.scheduler.WorkflowScheduler;
import org.example.core.factories.*;
import org.example.dataset.Dataset;
import org.example.dataset.DatasetSolution;
import org.example.sensors.TaskStateSensor;
import org.example.simulation.listeners.WorkflowBuffer;
import org.example.simulation.listeners.WorkflowCoordinator;
import org.example.simulation.listeners.UtilizationUpdater;
import org.example.core.entities.DynamicDatacenterBroker;
import org.example.core.registries.CloudletRegistry;
import org.example.core.registries.HostRegistry;
import org.example.simulation.listeners.WorkflowSubmitter;

import java.util.Calendar;

/// Represents the simulated world.
public class SimulatedWorld {
    private static final int NUM_USERS = 1;
    private static final boolean TRACE_FLAG = false;

    private final Dataset dataset;
    private final WorkflowSubmitter submitter;
    private final DynamicDatacenterBroker broker;

    @Builder
    public SimulatedWorld(@NonNull Dataset dataset,
                          @NonNull WorkflowReleaser releaser,
                          @NonNull WorkflowScheduler scheduler,
                          @NonNull WorkflowExecutor executor,
                          @NonNull SimulatedWorldConfig config) {
        this.dataset = dataset;

        // Create a CloudSimPlus object to initialize the simulation.
        CloudSim.init(NUM_USERS, Calendar.getInstance(), TRACE_FLAG);

        // Create factories for entities.
        var hostFactory = HostFactory.builder().build();
        var datacenterFactory = DatacenterFactory.builder().build();
        var vmFactory = VmFactory.builder().build();
        var brokerFactory = DatacenterBrokerFactory.builder().build();
        var cloudletFactory = CloudletFactory.builder().build();

        // Create entities.
        this.broker = brokerFactory.createBroker();
        var hosts = hostFactory.createHosts(dataset.getHosts());
        var vms = vmFactory.createVms(broker.getId(), dataset.getVms());
        var ignoredDc = datacenterFactory.createDatacenter(hosts);

        // Submits the VM list to the broker
        broker.submitGuestList(vms);

        CloudSim.terminateSimulation(config.getSimulationDuration());

        // Create tick listeners
        var coordinator = WorkflowCoordinator.builder()
                .broker(broker).cloudletFactory(cloudletFactory)
                .releaser(releaser).scheduler(scheduler).executor(executor).build();
        var buffer = WorkflowBuffer.builder()
                .coordinator(coordinator).releaser(releaser).build();
        submitter = WorkflowSubmitter.builder()
                .buffer(buffer).workflows(dataset.getWorkflows()).build();
        var ignoredUtilUpdater = UtilizationUpdater.builder()
                .monitoringUpdateInterval(config.getMonitoringUpdateInterval()).build();
    }

    /// Starts the simulation and waits all cloudlets to be executed,
    /// automatically stopping when there is no more events to process
    public DatasetSolution runSimulation() {
        CloudSim.startSimulation();
        CloudSim.stopSimulation();

        var cloudletRegistry = CloudletRegistry.getInstance();
        var hostRegistry = HostRegistry.getInstance();
        var taskStateSensor = TaskStateSensor.getInstance();

        // Prints the results when the simulation is over
        cloudletRegistry.printSummaryTable();
        System.err.printf("Total makespan (s)           : %.5f%n", cloudletRegistry.getTotalMakespan());
        System.err.printf("Total power consumption (W)  : %.2f%n", hostRegistry.getTotalPowerConsumptionW());
        System.err.printf("Total allocated VMs          : %d / %d%n", hostRegistry.getTotalAllocatedVms(), broker.getGuestList().size());
        System.err.printf("Unfinished Cloudlets         : %d / %d%n", cloudletRegistry.getRunningCloudletCount(), cloudletRegistry.getSize());
        System.err.printf("Total Cloudlet length (MI)   : %d%n", cloudletRegistry.getTotalCloudletLength());
        System.err.printf("Last task finish time (s)    : %.2f%n", cloudletRegistry.getLastCloudletFinishedAt());

        System.err.printf("Buffered Tasks               : %d%n", taskStateSensor.getBufferedTasks());
        System.err.printf("Released Tasks               : %d%n", taskStateSensor.getReleasedTasks());
        System.err.printf("Scheduled Tasks              : %d%n", taskStateSensor.getScheduledTasks());
        System.err.printf("Executed Tasks               : %d%n", taskStateSensor.getExecutedTasks());
        System.err.printf("Finished Tasks               : %d%n", taskStateSensor.getCompletedTasks());

        var solutionDataset = new Dataset(submitter.getSubmittedWorkflows(), dataset.getVms(), dataset.getHosts());
        return new DatasetSolution(solutionDataset, cloudletRegistry.getVmAssignments());
    }
}
