package org.example.api.scheduler.gym.algorithms;

import lombok.NonNull;
import org.example.api.dtos.TaskDto;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;
import org.example.api.scheduler.gym.GymEnvironment;
import org.example.api.scheduler.gym.GymSharedQueue;
import org.example.api.scheduler.gym.types.AgentResult;
import org.example.api.scheduler.gym.types.StaticAction;
import org.example.api.scheduler.gym.types.StaticObservation;
import org.example.api.scheduler.internal.algorithms.StaticSchedulingAlgorithm;

import java.util.*;

/// A scheduling algorithm that delegates the scheduling to external gym environment.
public class StaticGymSchedulingAlgorithm implements StaticSchedulingAlgorithm {
    private final GymEnvironment<StaticObservation, StaticAction> environment;

    public StaticGymSchedulingAlgorithm(@NonNull GymSharedQueue<StaticObservation, StaticAction> queue) {
        this.environment = new GymEnvironment<>(queue);
    }

    @Override
    public List<VmAssignmentDto> schedule(@NonNull List<TaskDto> tasks, @NonNull List<VmDto> vms) {
        var observation = new StaticObservation(tasks, vms);
        var action = environment.step(AgentResult.reward(observation, 0));
        return action.getAssignments();
    }
}
