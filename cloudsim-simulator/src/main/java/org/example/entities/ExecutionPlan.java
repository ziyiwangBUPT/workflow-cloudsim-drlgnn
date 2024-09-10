package org.example.entities;

import org.cloudbus.cloudsim.Vm;

import java.util.*;

public class ExecutionPlan {
    private final Map<Integer, Queue<WorkflowCloudlet>> vmTaskQueues;
    private final Map<Integer, Integer> pendingDependencies;
    private final Set<Integer> submittedTasks;

    public ExecutionPlan() {
        this.vmTaskQueues = new HashMap<>();
        this.pendingDependencies = new HashMap<>();
        this.submittedTasks = new HashSet<>();
    }

    public void freeFinishedTasks() {
        for (var vmId : vmTaskQueues.keySet()) {
            if (!vmTaskQueues.get(vmId).isEmpty()) {
                var task = vmTaskQueues.get(vmId).peek();
                if (task != null && task.isFinished()) {
                    vmTaskQueues.get(vmId).poll();
                    for (var childId : task.getChildIds()) {
                        pendingDependencies.put(childId, pendingDependencies.get(childId) - 1);
                    }
                }
            }
        }
    }

    public void submitReadyTasks(MonitoredDatacenterBroker broker) {
        for (var vmId : vmTaskQueues.keySet()) {
            if (!vmTaskQueues.get(vmId).isEmpty()) {
                var task = vmTaskQueues.get(vmId).peek();
                if (task != null
                        // Check if the task is not already running
                        && !submittedTasks.contains(task.getCloudletId())
                        // Check if all dependencies are resolved
                        && pendingDependencies.getOrDefault(task.getCloudletId(), 0) == 0) {
                    broker.submitCloudletList(List.of(task));
                    submittedTasks.add(task.getCloudletId());
                }
            }
        }
    }

    public void merge(ExecutionPlan graph) {
        // Merge nodes
        for (var vmId : graph.vmTaskQueues.keySet()) {
            if (vmTaskQueues.containsKey(vmId)) {
                vmTaskQueues.get(vmId).addAll(graph.vmTaskQueues.get(vmId));
            } else {
                vmTaskQueues.put(vmId, graph.vmTaskQueues.get(vmId));
            }
        }
        pendingDependencies.putAll(graph.pendingDependencies);
    }

    public void addTaskToVm(WorkflowCloudlet task, Vm vm) {
        task.setGuestId(vm.getId());
        if (!vmTaskQueues.containsKey(vm.getId())) {
            vmTaskQueues.put(vm.getId(), new ArrayDeque<>());
        }
        vmTaskQueues.get(vm.getId()).add(task);
        for (var childId : task.getChildIds()) {
            pendingDependencies.put(childId, pendingDependencies.getOrDefault(childId, 0) + 1);
        }
    }
}
