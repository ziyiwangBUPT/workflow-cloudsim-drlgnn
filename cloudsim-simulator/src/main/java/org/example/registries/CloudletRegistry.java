package org.example.registries;

import lombok.Getter;
import org.cloudbus.cloudsim.Cloudlet;
import org.example.tables.CloudletTable;

import java.util.ArrayList;
import java.util.List;

public class CloudletRegistry {
    @Getter
    private static final CloudletRegistry instance = new CloudletRegistry();

    private final List<Cloudlet> cloudlets = new ArrayList<>();

    private CloudletRegistry() {
    }

    public void registerNewCloudlets(List<Cloudlet> cloudlets) {
        this.cloudlets.addAll(cloudlets);
    }

    public int getCloudletCount() {
        return cloudlets.size();
    }

    public long getRunningCloudletCount() {
        return cloudlets.stream()
                .filter(c -> Cloudlet.CloudletStatus.INEXEC.equals(c.getStatus()))
                .count();
    }

    public long getTotalCloudletLength() {
        return cloudlets.stream()
                .mapToLong(Cloudlet::getCloudletLength)
                .sum();
    }

    public double getTotalMakespan() {
        return cloudlets.stream()
                .filter(c -> Cloudlet.CloudletStatus.SUCCESS.equals(c.getStatus()))
                .mapToDouble(c -> (c.getExecFinishTime() - c.getExecStartTime()))
                .sum();
    }

    public double getLastCloudletFinishedAt() {
        return cloudlets.stream()
                .filter(c -> Cloudlet.CloudletStatus.SUCCESS.equals(c.getStatus()))
                .mapToDouble(Cloudlet::getExecFinishTime)
                .max().orElse(0);
    }

    public void printSummaryTable() {
        new CloudletTable(cloudlets).print();
    }
}
