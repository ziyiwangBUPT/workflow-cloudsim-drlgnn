package org.example.registries;

import lombok.Getter;
import lombok.NonNull;
import org.cloudbus.cloudsim.Cloudlet;
import org.example.utils.SummaryTable;

import java.util.List;

/// A singleton class that holds a list of Cloudlets.
public class CloudletRegistry extends AbstractRegistry<Cloudlet> {
    @Getter
    private static final CloudletRegistry instance = new CloudletRegistry();

    private CloudletRegistry() {
    }

    /// Registers a new list of cloudlets.
    public void registerNewCloudlets(@NonNull List<Cloudlet> newCloudlets) {
        newCloudlets.forEach(cloudlet -> register(cloudlet.getCloudletId(), cloudlet));
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
    public double getTotalMakespan() {
        return itemStream().filter(c -> Cloudlet.CloudletStatus.SUCCESS.equals(c.getStatus()))
                .mapToDouble(c -> (c.getExecFinishTime() - c.getExecStartTime())).sum();
    }

    /// Gets the finish time of the last completed Cloudlet.
    public double getLastCloudletFinishedAt() {
        return itemStream().filter(c -> Cloudlet.CloudletStatus.SUCCESS.equals(c.getStatus()))
                .mapToDouble(Cloudlet::getExecFinishTime).max().orElse(0);
    }

    @Override
    protected SummaryTable<Cloudlet> buildSummaryTable() {
        var summaryTable = new SummaryTable<Cloudlet>();

        summaryTable.addColumn("Cloudlet", SummaryTable.ID_UNIT, SummaryTable.STRING_FORMAT, Cloudlet::getCloudletId);
        summaryTable.addColumn("Guest", SummaryTable.ID_UNIT, SummaryTable.STRING_FORMAT, Cloudlet::getGuestId);
        summaryTable.addColumn("    Status    ", SummaryTable.ID_UNIT, SummaryTable.STRING_FORMAT, cloudlet -> cloudlet.getStatus().name());
        summaryTable.addColumn("Cloudlet Len", SummaryTable.MI_UNIT, SummaryTable.INTEGER_FORMAT, Cloudlet::getCloudletLength);
        summaryTable.addColumn("Finished Len", SummaryTable.MI_UNIT, SummaryTable.INTEGER_FORMAT, Cloudlet::getCloudletFinishedSoFar);
        summaryTable.addColumn("Start Time", SummaryTable.S_UNIT, SummaryTable.DECIMAL_FORMAT, Cloudlet::getExecStartTime);
        summaryTable.addColumn("End Time  ", SummaryTable.S_UNIT, SummaryTable.DECIMAL_FORMAT, Cloudlet::getExecFinishTime);
        summaryTable.addColumn("Makespan  ", SummaryTable.S_UNIT, SummaryTable.DECIMAL_FORMAT, cloudlet -> cloudlet.isFinished() ? cloudlet.getExecFinishTime() - cloudlet.getExecStartTime() : 0);

        return summaryTable;
    }
}
