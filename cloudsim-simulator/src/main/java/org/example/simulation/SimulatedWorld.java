package org.example.simulation;

import org.cloudbus.cloudsim.core.CloudSim;
import org.example.entities.ExecutionPlan;
import org.example.dataset.Dataset;
import org.example.schedulers.RoundRobinScheduler;
import org.example.ticks.MonitoredHostsUpdater;
import org.example.factories.*;
import org.example.entities.DynamicDatacenterBroker;
import org.example.ticks.WorkflowExecutor;
import org.example.ticks.WorkflowScheduler;
import org.example.registries.CloudletRegistry;
import org.example.registries.HostRegistry;

import java.util.Calendar;

/// Represents the simulated world.
public class SimulatedWorld {
    private static final int NUM_USERS = 1;
    private static final boolean TRACE_FLAG = false;

    private final DynamicDatacenterBroker broker;

    public SimulatedWorld(Dataset dataset, SimulatedWorldConfig config) {
        // Create a CloudSimPlus object to initialize the simulation.
        CloudSim.init(NUM_USERS, Calendar.getInstance(), TRACE_FLAG);

        // Create factories for entities.
        var hostFactory = HostFactory.builder().build();
        var datacenterFactory = DatacenterFactory.builder().build();
        var vmFactory = VmFactory.builder().build();
        var brokerFactory = DatacenterBrokerFactory.builder().build();
        var cloudletFactory = CloudletFactory.builder().build();

        // Create entities.
        broker = brokerFactory.createBroker();
        var hosts = hostFactory.createHosts(dataset.getHosts());
        var vms = vmFactory.createVms(broker.getId(), dataset.getVms());
        var _ = datacenterFactory.createDatacenter(hosts);

        // Submits the VM list to the broker
        broker.submitGuestList(vms);

        var executionPlan = new ExecutionPlan();
        CloudSim.terminateSimulation(config.getSimulationDuration());
        WorkflowScheduler.builder()
                .broker(broker)
                .scheduler(new RoundRobinScheduler())
                .executionPlan(executionPlan)
                .cloudletFactory(cloudletFactory)
                .schedulingInterval(config.getSchedulingInterval())
                .datasetWorkflows(dataset.getWorkflows()).build();
        WorkflowExecutor.builder()
                .broker(broker)
                .executionPlan(executionPlan).build();
        MonitoredHostsUpdater.builder()
                .monitoringUpdateInterval(config.getMonitoringUpdateInterval()).build();
    }

    public void runSimulation() {
        // Starts the simulation and waits all cloudlets to be executed,
        // automatically stopping when there is no more events to process
        CloudSim.startSimulation();
        CloudSim.stopSimulation();

        var cloudletRegistry = CloudletRegistry.getInstance();
        var hostRegistry = HostRegistry.getInstance();

        // Prints the results when the simulation is over
        cloudletRegistry.printSummaryTable();
        System.out.printf("Total makespan (s)           : %.5f%n", cloudletRegistry.getTotalMakespan());
        System.out.printf("Total power consumption (W)  : %.2f%n", hostRegistry.getTotalPowerConsumptionW());
        System.out.printf("Total allocated VMs          : %d / %d%n", hostRegistry.getTotalAllocatedVms(), broker.getGuestList().size());
        System.out.printf("Unfinished Cloudlets         : %d / %d%n", cloudletRegistry.getRunningCloudletCount(), cloudletRegistry.getCloudletCount());
        System.out.printf("Total Cloudlet length (MI)   : %d%n", cloudletRegistry.getTotalCloudletLength());
        System.out.printf("Last task finish time (s)    : %.2f%n", cloudletRegistry.getLastCloudletFinishedAt());
    }
}
