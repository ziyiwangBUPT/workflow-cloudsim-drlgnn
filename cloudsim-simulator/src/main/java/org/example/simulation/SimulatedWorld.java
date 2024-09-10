package org.example.simulation;

import org.cloudbus.cloudsim.core.CloudSim;
import org.example.entities.ExecutionPlan;
import org.example.models.Dataset;
import org.example.schedulers.RoundRobinScheduler;
import org.example.ticks.MonitoredHostsUpdater;
import org.example.factories.*;
import org.example.entities.MonitoredDatacenterBroker;
import org.example.ticks.WorkflowExecutor;
import org.example.ticks.WorkflowScheduler;
import org.example.registries.CloudletRegistry;
import org.example.registries.HostRegistry;

import java.util.Calendar;

public class SimulatedWorld {
    private static final int NUM_USERS = 1;
    private static final boolean TRACE_FLAG = false;

    private final MonitoredDatacenterBroker broker;

    public SimulatedWorld(Dataset dataset, SimulatedWorldConfig config) {
        // Creates a CloudSimPlus object to initialize the simulation.
        CloudSim.init(NUM_USERS, Calendar.getInstance(), TRACE_FLAG);

        var hostFactory = HostFactory.builder().build();
        var datacenterFactory = DatacenterFactory.builder().build();
        var vmFactory = VmFactory.builder().build();
        var brokerFactory = DatacenterBrokerFactory.builder().build();
        var cloudletFactory = CloudletFactory.builder().build();

        // Creates entities
        broker = brokerFactory.createBroker();
        var hosts = hostFactory.createHosts(dataset.getHosts());
        var vms = vmFactory.createVms(broker.getId(), dataset.getVms(), hosts);
        var _ = datacenterFactory.createDatacenter(hosts);

        // Submits the VM list to the broker
        broker.submitGuestList(vms);

        var executionPlan = new ExecutionPlan();
        CloudSim.terminateSimulation(config.getSimulationDurationSeconds());
        WorkflowScheduler.builder()
                .broker(broker)
                .scheduler(new RoundRobinScheduler())
                .executionPlan(executionPlan)
                .cloudletFactory(cloudletFactory)
                .schedulingInterval(config.getSchedulingIntervalSeconds())
                .datasetWorkflows(dataset.getWorkflows()).build();
        WorkflowExecutor.builder()
                .broker(broker)
                .executionPlan(executionPlan).build();
        MonitoredHostsUpdater.builder()
                .monitoringUpdateInterval(config.getMonitoringUpdateIntervalSeconds()).build();
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
