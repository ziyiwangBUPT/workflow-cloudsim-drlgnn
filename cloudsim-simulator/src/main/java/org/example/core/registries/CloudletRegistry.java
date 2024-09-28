package org.example.core.registries;

import lombok.Getter;
import lombok.NonNull;
import org.cloudbus.cloudsim.Cloudlet;
import org.example.dataset.DatasetVmAssignment;
import org.example.dataset.DatasetTask;
import org.example.utils.SummaryTable;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/// A singleton class that holds a list of Cloudlets.
public class CloudletRegistry extends AbstractRegistry<Cloudlet> {
    @Getter
    private static final CloudletRegistry instance = new CloudletRegistry();

    // Cloudlet ID -> (Workflow ID + Task ID)
    private final Map<Integer, WorkflowTaskId> cloudletMap = new HashMap<>();

    private CloudletRegistry() {
    }

    /// Registers a new list of cloudlets.
    public void registerNewCloudlet(@NonNull Cloudlet newCloudlet, @NonNull DatasetTask mappedTask) {
        register(newCloudlet.getCloudletId(), newCloudlet);
        cloudletMap.put(newCloudlet.getCloudletId(), new WorkflowTaskId(mappedTask.getWorkflowId(), mappedTask.getId()));
    }

    /// Gets the count of Cloudlets that are currently running.
    public long getRunningCloudletCount() {
        return itemStream().filter(c -> Cloudlet.CloudletStatus.INEXEC.equals(c.getStatus())).count();
    }

    /// Gets the total length of all Cloudlets.
    public long getTotalCloudletLength() {
        return itemStream().mapToLong(Cloudlet::getCloudletLength).sum();
    }

    /// Calculates the total makespan of completed Cloudlets
    /// This the difference between starting cloudlet and ending cloudlet start end times.
    public double getMakespan() {
        var minStartTime = itemStream().filter(Cloudlet::isFinished).mapToDouble(Cloudlet::getExecStartTime).min().orElse(0);
        var maxEndTime = itemStream().filter(Cloudlet::isFinished).mapToDouble(Cloudlet::getExecFinishTime).max().orElse(0);
        return maxEndTime - minStartTime;
    }

    /// Gets the finish time of the last completed Cloudlet.
    public double getLastCloudletFinishedAt() {
        return itemStream().filter(c -> Cloudlet.CloudletStatus.SUCCESS.equals(c.getStatus()))
                .mapToDouble(Cloudlet::getExecFinishTime).max().orElse(0);
    }

    public double getMakespanDelta(Cloudlet cloudlet) {
        if (!cloudlet.isFinished()) {
            return 0;
        }

        var vmRegistry = VmRegistry.getInstance();
        var actualMakespan = cloudlet.getExecFinishTime() - cloudlet.getExecStartTime();
        var estMakespan = vmRegistry.estimateMakespan(cloudlet.getGuestId(), cloudlet.getCloudletLength());
        return actualMakespan - estMakespan;
    }

    @Override
    protected SummaryTable<Cloudlet> buildSummaryTable() {
        var summaryTable = new SummaryTable<Cloudlet>();

        summaryTable.addColumn("Cloudlet", SummaryTable.ID_UNIT, SummaryTable.STRING_FORMAT, Cloudlet::getCloudletId);
        summaryTable.addColumn("Workflow", SummaryTable.ID_UNIT, SummaryTable.STRING_FORMAT, cloudlet -> cloudletMap.get(cloudlet.getCloudletId()).workflowId());
        summaryTable.addColumn("Task    ", SummaryTable.ID_UNIT, SummaryTable.STRING_FORMAT, cloudlet -> cloudletMap.get(cloudlet.getCloudletId()).taskId());
        summaryTable.addColumn("Guest   ", SummaryTable.ID_UNIT, SummaryTable.STRING_FORMAT, Cloudlet::getGuestId);
        summaryTable.addColumn("    Status    ", SummaryTable.ID_UNIT, SummaryTable.STRING_FORMAT, cloudlet -> cloudlet.getStatus().name());
        summaryTable.addColumn("Cloudlet Len", SummaryTable.MI_UNIT, SummaryTable.INTEGER_FORMAT, Cloudlet::getCloudletLength);
        summaryTable.addColumn("Finished Len", SummaryTable.MI_UNIT, SummaryTable.INTEGER_FORMAT, Cloudlet::getCloudletFinishedSoFar);
        summaryTable.addColumn("Start Time", SummaryTable.S_UNIT, SummaryTable.DECIMAL_FORMAT, Cloudlet::getExecStartTime);
        summaryTable.addColumn("End Time  ", SummaryTable.S_UNIT, SummaryTable.DECIMAL_FORMAT, Cloudlet::getExecFinishTime);
        summaryTable.addColumn("Makespan  ", SummaryTable.S_UNIT, SummaryTable.DECIMAL_FORMAT, cloudlet -> cloudlet.isFinished() ? cloudlet.getExecFinishTime() - cloudlet.getExecStartTime() : 0);
        summaryTable.addColumn("Makespan Delta", SummaryTable.S_UNIT, SummaryTable.DECIMAL_FORMAT, this::getMakespanDelta);

        return summaryTable;
    }

    public List<DatasetVmAssignment> getVmAssignments() {
        return itemStream().map(cloudlet -> DatasetVmAssignment.builder()
                        .workflowId(cloudletMap.get(cloudlet.getCloudletId()).workflowId())
                        .taskId(cloudletMap.get(cloudlet.getCloudletId()).taskId())
                        .vmId(cloudlet.getGuestId())
                        .startTime(cloudlet.getExecStartTime())
                        .endTime(cloudlet.getExecFinishTime())
                        .build())
                .toList();
    }

    /// Private record classes to hold Workflow Task ID.
    private record WorkflowTaskId(int workflowId, int taskId) {
    }
}
