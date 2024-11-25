package org.example.core.factories;

import lombok.Builder;
import lombok.NonNull;
import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.UtilizationModelFull;
import org.example.dataset.DatasetTask;
import org.example.core.registries.CloudletRegistry;

import java.util.concurrent.atomic.AtomicInteger;


/// Factory for creating Cloudlets.
@Builder
public class CloudletFactory {
    private static final AtomicInteger CLOUDLET_ID = new AtomicInteger(0);

    @Builder.Default
    private final int cloudletFileSize = 300;
    @Builder.Default
    private final int cloudletOutputSize = 300;

    /// Create a Cloudlet from a DatasetTask.
    public Cloudlet createCloudlet(int brokerId, @NonNull DatasetTask datasetTask) {
        try {
            var cloudletId = CLOUDLET_ID.getAndIncrement();
            var cloudletLength = datasetTask.getLength();
            var pesNumber = 1; // We assume all cloudlets have 1 PE
            var utilizationModel = new UtilizationModelFull();

            // Create cloudlet
            var cloudlet = new Cloudlet(cloudletId, cloudletLength, pesNumber, cloudletFileSize, cloudletOutputSize,
                    utilizationModel, utilizationModel, utilizationModel);
            cloudlet.setUserId(brokerId);

            // Register cloudlet
            var cloudletRegistry = CloudletRegistry.getInstance();
            cloudletRegistry.registerNewCloudlet(cloudlet, datasetTask);

            return cloudlet;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
