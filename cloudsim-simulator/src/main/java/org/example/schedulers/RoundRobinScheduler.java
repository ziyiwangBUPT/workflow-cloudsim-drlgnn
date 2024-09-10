package org.example.schedulers;

import org.cloudbus.cloudsim.Vm;
import org.example.entities.ExecutionPlan;
import org.example.entities.WorkflowCloudlet;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.List;

public class RoundRobinScheduler implements Scheduler {
    @Override
    public ExecutionPlan schedule(List<List<WorkflowCloudlet>> workflows, List<Vm> vms) {
        var executionPlan = new ExecutionPlan();
        if (workflows.isEmpty() || vms.isEmpty()) {
            return executionPlan;
        }

        var readyQueue = new ArrayDeque<WorkflowCloudlet>();
        for (var workflow : workflows) {
            for (var task : workflow) {
                if (task.isStartNode()) {
                    readyQueue.add(task);
                }
            }
        }

        var taskMap = new HashMap<Integer, WorkflowCloudlet>();
        for (var workflow : workflows) {
            for (var task : workflow) {
                taskMap.put(task.getCloudletId(), task);
            }
        }

        var vmIndex = 0;
        while (!readyQueue.isEmpty()) {
            var task = readyQueue.removeFirst();

            var vm = vms.get((vmIndex++) % vms.size());
            // Check if the cloudlet is assignable to the VM
            while (vm.getNumberOfPes() < task.getNumberOfPes() || vm.getSize() < task.getCloudletFileSize()) {
                vm = vms.get((vmIndex++) % vms.size());
            }

            executionPlan.addTaskToVm(task, vm);
            for (var childId : task.getChildIds()) {
                readyQueue.add(taskMap.get(childId));
            }
        }

        return executionPlan;
    }
}
