package org.example;

import org.example.dataset.Dataset;
import org.example.simulation.SimulatedWorld;
import org.example.simulation.SimulatedWorldConfig;

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        // Read input from stdin
        var scanner = new Scanner(System.in);
        var dataset = Dataset.fromJson(scanner.nextLine());

        // Configure simulation
        var config = SimulatedWorldConfig.builder()
                .simulationDuration(1000)
                .schedulingInterval(10)
                .monitoringUpdateInterval(5)
                .build();

        // Run simulation
        var world = new SimulatedWorld(dataset, config);
        world.runSimulation();
    }
}
