package org.example.simulation;

import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import org.cloudbus.cloudsim.core.CloudSim;
import org.example.api.executor.WorkflowExecutor;
import org.example.api.scheduler.WorkflowScheduler;
import org.example.core.factories.*;
import org.example.dataset.Dataset;
import org.example.dataset.DatasetSolution;
import org.example.simulation.listeners.WorkflowCoordinator;
import org.example.simulation.listeners.UtilizationUpdater;
import org.example.core.entities.DynamicDatacenterBroker;
import org.example.core.registries.CloudletRegistry;
import org.example.simulation.listeners.WorkflowSubmitter;

import java.util.Calendar;

/// Represents the simulated world.
public class SimulatedWorld {
    private static final int NUM_USERS = 1;
    private static final boolean TRACE_FLAG = false;

    private final Dataset dataset;
    private final WorkflowSubmitter submitter;

    @Getter
    private final DynamicDatacenterBroker broker;

    @Builder
    public SimulatedWorld(@NonNull Dataset dataset,
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
        var ignoredDc = datacenterFactory.createDatacenter(hosts, dataset.getVms());

        // Submits the VM list to the broker
        broker.submitGuestList(vms);

        CloudSim.terminateSimulation(config.getSimulationDuration());

        // Create tick listeners
        var coordinator = WorkflowCoordinator.builder()
                .broker(broker).cloudletFactory(cloudletFactory)
                .scheduler(scheduler).executor(executor).build();
        submitter = WorkflowSubmitter.builder()
                .coordinator(coordinator).workflows(dataset.getWorkflows()).build();
        var ignoredUtilUpdater = UtilizationUpdater.builder()
                .monitoringUpdateInterval(config.getMonitoringUpdateInterval()).build();
    }

    /// Starts the simulation and waits all cloudlets to be executed,
    /// automatically stopping when there is no more events to process
    public DatasetSolution runSimulation() {
        CloudSim.startSimulation();
        CloudSim.stopSimulation();

        var cloudletRegistry = CloudletRegistry.getInstance();
        var solutionDataset = new Dataset(submitter.getSubmittedWorkflows(), dataset.getVms(), dataset.getHosts());
        return new DatasetSolution(solutionDataset, cloudletRegistry.getVmAssignments());
    }
}
