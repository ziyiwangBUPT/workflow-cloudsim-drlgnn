package org.example.simulation;

import lombok.NonNull;
import org.cloudbus.cloudsim.core.CloudSim;
import org.example.api.impl.LocalWorkflowExecutor;
import org.example.api.impl.PeriodicWorkflowReleaser;
import org.example.api.impl.RoundRobinWorkflowScheduler;
import org.example.dataset.Dataset;
import org.example.dataset.DatasetSolution;
import org.example.ticks.WorkflowBuffer;
import org.example.ticks.WorkflowCoordinator;
import org.example.ticks.UtilizationUpdater;
import org.example.factories.*;
import org.example.entities.DynamicDatacenterBroker;
import org.example.registries.CloudletRegistry;
import org.example.registries.HostRegistry;
import org.example.ticks.WorkflowSubmitter;

import java.util.Calendar;

/// Represents the simulated world.
public class SimulatedWorld {
    private static final int NUM_USERS = 1;
    private static final boolean TRACE_FLAG = false;

    private final Dataset dataset;
    private final DynamicDatacenterBroker broker;

    public SimulatedWorld(@NonNull Dataset dataset, @NonNull SimulatedWorldConfig config) {
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
        var _ = datacenterFactory.createDatacenter(hosts);

        // Submits the VM list to the broker
        broker.submitGuestList(vms);

        var releaser = new PeriodicWorkflowReleaser();
        var scheduler = new RoundRobinWorkflowScheduler();
        var executor = new LocalWorkflowExecutor();

        CloudSim.terminateSimulation(config.getSimulationDuration());

        // Create tick listeners
        var coordinator = WorkflowCoordinator.builder()
                .broker(broker).cloudletFactory(cloudletFactory)
                .releaser(releaser).scheduler(scheduler).executor(executor).build();
        var buffer = WorkflowBuffer.builder()
                .coordinator(coordinator).releaser(releaser).build();
        var _ = WorkflowSubmitter.builder()
                .buffer(buffer).workflows(dataset.getWorkflows()).build();
        var _ = UtilizationUpdater.builder()
                .monitoringUpdateInterval(config.getMonitoringUpdateInterval()).build();
    }

    /// Starts the simulation and waits all cloudlets to be executed,
    /// automatically stopping when there is no more events to process
    public DatasetSolution runSimulation() {
        CloudSim.startSimulation();
        CloudSim.stopSimulation();

        var cloudletRegistry = CloudletRegistry.getInstance();
        var hostRegistry = HostRegistry.getInstance();

        // Prints the results when the simulation is over
        cloudletRegistry.printSummaryTable();
        System.err.printf("Total makespan (s)           : %.5f%n", cloudletRegistry.getTotalMakespan());
        System.err.printf("Total power consumption (W)  : %.2f%n", hostRegistry.getTotalPowerConsumptionW());
        System.err.printf("Total allocated VMs          : %d / %d%n", hostRegistry.getTotalAllocatedVms(), broker.getGuestList().size());
        System.err.printf("Unfinished Cloudlets         : %d / %d%n", cloudletRegistry.getRunningCloudletCount(), cloudletRegistry.getSize());
        System.err.printf("Total Cloudlet length (MI)   : %d%n", cloudletRegistry.getTotalCloudletLength());
        System.err.printf("Last task finish time (s)    : %.2f%n", cloudletRegistry.getLastCloudletFinishedAt());

        return new DatasetSolution(dataset, cloudletRegistry.getVmAssignments());
    }
}
