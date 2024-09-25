package org.example.api.scheduler.impl;

import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;
import org.example.api.scheduler.StaticWorkflowScheduler;

import java.util.ArrayList;
import java.util.List;

/// The round-robin workflow scheduler.
public class RoundRobinWorkflowScheduler extends StaticWorkflowScheduler {
    @Override
    protected List<VmAssignmentDto> schedule(List<WorkflowDto> workflows, List<VmDto> vms) {
        System.out.println("Scheduling workflows using Round Robin...");
        System.out.println("Number of workflows: " + workflows.size());
        System.out.println("Number of VMs: " + vms.size());

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
