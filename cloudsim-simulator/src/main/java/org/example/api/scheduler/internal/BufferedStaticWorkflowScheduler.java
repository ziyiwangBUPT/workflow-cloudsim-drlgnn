package org.example.api.scheduler.internal;

import lombok.NonNull;
import org.cloudbus.cloudsim.core.CloudSim;
import org.example.api.dtos.TaskDto;
import org.example.api.dtos.VmAssignmentDto;
import org.example.api.dtos.VmDto;
import org.example.api.dtos.WorkflowDto;
import org.example.api.scheduler.WorkflowScheduler;
import org.example.api.scheduler.internal.algorithms.StaticSchedulingAlgorithm;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/// Dynamic workflow scheduler that uses a static scheduler with buffer underneath.
public class BufferedStaticWorkflowScheduler implements WorkflowScheduler {
    private final StaticSchedulingAlgorithm algorithm;

    private final int bufferSize;
    private final List<VmDto> vms = new ArrayList<>();
    private final List<WorkflowDto> bufferedWorkflows = new ArrayList<>();
    private final List<VmAssignmentDto> lastSchedulingResult = new ArrayList<>();
    private double lastReleaseTime = 0;

    public BufferedStaticWorkflowScheduler(int bufferSize, @NonNull StaticSchedulingAlgorithm algorithm) {
        this.algorithm = algorithm;
        this.bufferSize = bufferSize;
    }

    @Override
    public void notifyNewVm(@NonNull VmDto newVm) {
        vms.add(newVm);
    }

    @Override
    public void notifyNewWorkflow(@NonNull WorkflowDto newWorkflow) {
        bufferedWorkflows.add(newWorkflow);
    }

    @Override
    public Optional<VmAssignmentDto> schedule() {
        if (!lastSchedulingResult.isEmpty()) {
            return Optional.ofNullable(lastSchedulingResult.removeFirst());
        }
        if (bufferedWorkflows.isEmpty()) {
            return Optional.empty();
        }

        var bufferedTaskCount = bufferedWorkflows.stream().mapToInt(w -> w.getTasks().size()).sum();
        if (bufferedTaskCount < bufferSize) {
            if (CloudSim.clock() - lastReleaseTime > 100) {
                // Release all inside buffer if timed out
                var tasks = new ArrayList<TaskDto>();
                bufferedWorkflows.forEach(w -> tasks.addAll(w.getTasks()));
                bufferedWorkflows.clear();
                lastSchedulingResult.addAll(algorithm.schedule(tasks, vms));
                lastReleaseTime = CloudSim.clock();
                return Optional.of(lastSchedulingResult.removeFirst());
            }

            // We can wait a bit more...
            return Optional.empty();
        }

        var tasks = new ArrayList<TaskDto>();
        while (!bufferedWorkflows.isEmpty()) {
            var tasksInNextWorkflow = bufferedWorkflows.getFirst().getTasks().size();
            if (tasks.size() + tasksInNextWorkflow > bufferSize) {
                // Next workflow will exceed max size
                break;
            }
            tasks.addAll(bufferedWorkflows.removeFirst().getTasks());
        }
        lastSchedulingResult.addAll(algorithm.schedule(tasks, vms));
        lastReleaseTime = CloudSim.clock();
        return Optional.of(lastSchedulingResult.removeFirst());
    }
}
