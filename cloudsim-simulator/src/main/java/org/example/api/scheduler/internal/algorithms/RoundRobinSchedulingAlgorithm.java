package org.example.api.scheduler.internal.algorithms;

import lombok.NonNull;
import org.example.api.dtos.TaskDto;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;

import java.util.ArrayList;
import java.util.List;

/// The round-robin workflow scheduler.
public class RoundRobinSchedulingAlgorithm implements StaticSchedulingAlgorithm {
    @Override
    public List<VmAssignmentDto> schedule(@NonNull List<TaskDto> tasks, @NonNull List<VmDto> vms) {
        var vmIndex = 0;
        var schedulingResult = new ArrayList<VmAssignmentDto>();

        for (var task : tasks) {
            var vm = vms.get(vmIndex);
            vmIndex = (vmIndex + 1) % vms.size();
            while (!vm.canRunTask(task)) {
                vm = vms.get(vmIndex);
                vmIndex = (vmIndex + 1) % vms.size();
            }

            var assignment = new VmAssignmentDto(vm.getId(), task.getWorkflowId(), task.getId());
            schedulingResult.add(assignment);
        }

        return schedulingResult;
    }
}
