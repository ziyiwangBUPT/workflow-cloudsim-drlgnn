package org.example.registries;

import lombok.Getter;
import lombok.NonNull;
import org.cloudbus.cloudsim.Cloudlet;
import org.example.tables.CloudletTable;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/// A singleton class that holds a list of Cloudlets.
public class CloudletRegistry {
    @Getter
    private static final CloudletRegistry instance = new CloudletRegistry();

    private final Map<CloudletId, Cloudlet> cloudlets = new HashMap<>();

    private CloudletRegistry() {
    }

    /// Registers a new list of cloudlets.
    public void registerNewCloudlets(@NonNull List<Cloudlet> newCloudlets) {
        newCloudlets.forEach(cloudlet -> cloudlets.put(new CloudletId(cloudlet.getCloudletId()), cloudlet));
    }

    /// Gets the total count of registered Cloudlets,
    public int getCloudletCount() {
        return cloudlets.size();
    }

    /// Gets the count of Cloudlets that are currently running.
    public long getRunningCloudletCount() {
        return cloudlets.values().stream().filter(c -> Cloudlet.CloudletStatus.INEXEC.equals(c.getStatus())).count();
    }

    /// Gets the total length of all Cloudlets.
    public long getTotalCloudletLength() {
        return cloudlets.values().stream().mapToLong(Cloudlet::getCloudletLength).sum();
    }

    /// Calculates the total makespan of completed Cloudlets
    public double getTotalMakespan() {
        return cloudlets.values().stream().filter(c -> Cloudlet.CloudletStatus.SUCCESS.equals(c.getStatus()))
                .mapToDouble(c -> (c.getExecFinishTime() - c.getExecStartTime())).sum();
    }

    /// Gets the finish time of the last completed Cloudlet.
    public double getLastCloudletFinishedAt() {
        return cloudlets.values().stream().filter(c -> Cloudlet.CloudletStatus.SUCCESS.equals(c.getStatus()))
                .mapToDouble(Cloudlet::getExecFinishTime).max().orElse(0);
    }

    /// Prints a summary table of all Cloudlets.
    public void printSummaryTable() {
        new CloudletTable(cloudlets.values()).print();
    }

    /// A private record to represent a Cloudlet ID.
    private record CloudletId(int id) {
    }
}
