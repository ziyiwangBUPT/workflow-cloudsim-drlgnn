package org.example;

import com.google.gson.FieldNamingPolicy;
import com.google.gson.GsonBuilder;
import org.example.models.Dataset;
import org.example.simulation.SimulatedWorld;
import org.example.simulation.SimulatedWorldConfig;

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        var scanner = new Scanner(System.in);
        var input = scanner.nextLine();

        var gson = new GsonBuilder().setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES).create();
        var datasetDto = gson.fromJson(input, Dataset.class);

        var config = SimulatedWorldConfig.builder()
                .simulationDurationSeconds(1000) // Simulation runs for X seconds
                .schedulingIntervalSeconds(1) // Scheduling happens every X seconds (batch scheduling)
                .monitoringUpdateIntervalSeconds(5) // All host states are updated X seconds
                .build();

        var world = new SimulatedWorld(datasetDto, config);
        world.runSimulation();
    }
}
