package org.example.api.scheduler.gym.mappers;

import lombok.NonNull;
import org.example.api.dtos.TaskDto;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;
import org.example.api.scheduler.gym.GymEnvironment;
import org.example.api.scheduler.gym.types.AgentResult;
import org.example.api.scheduler.gym.types.Action;
import org.example.api.scheduler.gym.types.JsonObservation;
import org.example.utils.GsonHelper;

import java.util.List;
import java.util.Optional;

/// A simple gym workflow algorithm.
/// The observation is the JSON representation of the VMs and tasks.
/// The reward is always 0.
public class SimpleGymMapper implements GymWorkflowMapper<JsonObservation, Action> {
    @Override
    public Optional<VmAssignmentDto> schedule(@NonNull GymEnvironment<JsonObservation, Action> environment,
                                              @NonNull List<VmDto> vms, @NonNull List<TaskDto> tasks) {
        // Observation
        var gson = GsonHelper.getGson();
        var vmJson = gson.toJson(vms);
        var taskJson = gson.toJson(tasks);
        var observation = new JsonObservation(vmJson, taskJson);

        // Reward
        var reward = 0d;

        // Action
        var action = environment.step(AgentResult.reward(observation, reward));
        if (action.isNoOp()) {
            return Optional.empty();
        }

        var assignment = new VmAssignmentDto(action.getVmId(), action.getWorkflowId(), action.getTaskId());
        return Optional.of(assignment);
    }
}
