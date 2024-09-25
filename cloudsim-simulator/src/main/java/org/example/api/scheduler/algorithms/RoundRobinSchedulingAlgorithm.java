package org.example.api.scheduler.algorithms;

import lombok.NonNull;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;

import java.util.ArrayList;
import java.util.List;

/// The round-robin workflow scheduler.
public class RoundRobinSchedulingAlgorithm implements StaticSchedulingAlgorithm {
    @Override
    public List<VmAssignmentDto> schedule(@NonNull List<WorkflowDto> workflows, @NonNull List<VmDto> vms) {
        var vmIndex = 0;
        var schedulingResult = new ArrayList<VmAssignmentDto>();

        for (var workflow : workflows) {
            for (var task : workflow.getTasks()) {
                var vm = vms.get(vmIndex);
                vmIndex = (vmIndex + 1) % vms.size();
                while (!vm.canRunTask(task)) {
                    vm = vms.get(vmIndex);
                    vmIndex = (vmIndex + 1) % vms.size();
                }

                var assignment = new VmAssignmentDto(vm.getId(), workflow.getId(), task.getId());
                schedulingResult.add(assignment);
            }
        }

        return schedulingResult;
    }
}
