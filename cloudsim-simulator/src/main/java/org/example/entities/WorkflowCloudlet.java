package org.example.entities;

import lombok.Builder;
import lombok.Getter;
import org.cloudbus.cloudsim.Cloudlet;
import org.cloudbus.cloudsim.UtilizationModel;

import java.util.List;

@Getter
public class WorkflowCloudlet extends Cloudlet {
    private final List<Integer> childIds;
    private final boolean isStartNode;

    @Builder
    public WorkflowCloudlet(int cloudletId, List<Integer> childCloudletIds, boolean isStartNode, long cloudletLength, int pesNumber,
                            long cloudletFileSize, long cloudletOutputSize, UtilizationModel utilizationModelCpu, UtilizationModel utilizationModelRam, UtilizationModel utilizationModelBw) {
        super(cloudletId, cloudletLength, pesNumber, cloudletFileSize, cloudletOutputSize, utilizationModelCpu, utilizationModelRam, utilizationModelBw);
        this.childIds = childCloudletIds;
        this.isStartNode = isStartNode;
    }
}
