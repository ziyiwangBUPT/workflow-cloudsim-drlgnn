package org.example.factories;

import lombok.Builder;
import org.cloudbus.cloudsim.UtilizationModelFull;
import org.example.entities.WorkflowCloudlet;
import org.example.dataset.DatasetTask;
import org.example.dataset.DatasetWorkflow;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;


// TODO: Remove?
@Builder
public class CloudletFactory {
    private static final AtomicInteger CLOUDLET_ID = new AtomicInteger(0);

    @Builder.Default
    private final int cloudletFileSize = 300;
    @Builder.Default
    private final int cloudletOutputSize = 300;

    private int getCloudletId(Map<Integer, Integer> taskIdToCloudletId, int taskId) {
        if (!taskIdToCloudletId.containsKey(taskId)) {
            taskIdToCloudletId.put(taskId, CLOUDLET_ID.getAndIncrement());
        }
        return taskIdToCloudletId.get(taskId);
    }

    private WorkflowCloudlet createCloudlet(int brokerId, HashMap<Integer, Integer> taskIdToCloudletId, DatasetTask task) {
        try {
            // Map task IDs to cloudlet IDs
            var id = getCloudletId(taskIdToCloudletId, task.getId());
            var childIds = new ArrayList<Integer>();
            for (var i : task.getChildIds()) {
                var cloudletId = getCloudletId(taskIdToCloudletId, i);
                childIds.add(cloudletId);
            }

            // Create cloudlet
            var cloudlet = WorkflowCloudlet.builder()
                    .cloudletId(id)
                    .childCloudletIds(childIds)
                    .isStartNode(task.getId() == 0) // 0 is the start node
                    .cloudletLength(task.getLength())
                    .pesNumber(task.getReqCores())
                    .cloudletFileSize(cloudletFileSize)
                    .cloudletOutputSize(cloudletOutputSize)
                    .utilizationModelCpu(new UtilizationModelFull())
                    .utilizationModelRam(new UtilizationModelFull())
                    .utilizationModelBw(new UtilizationModelFull())
                    .build();
            cloudlet.setUserId(brokerId);
            return cloudlet;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private List<WorkflowCloudlet> createCloudlets(int brokerId, DatasetWorkflow datasetWorkflow) {
        var cloudlets = new ArrayList<WorkflowCloudlet>();
        var tasks = datasetWorkflow.getTasks();
        var taskIdToCloudletId = new HashMap<Integer, Integer>();
        for (var task : tasks) {
            var cloudlet = createCloudlet(brokerId, taskIdToCloudletId, task);
            cloudlets.add(cloudlet);
        }
        return cloudlets;
    }

    public List<List<WorkflowCloudlet>> createCloudlets(int brokerId, List<DatasetWorkflow> datasetWorkflows) {
        var cloudlets = new ArrayList<List<WorkflowCloudlet>>();
        for (var datasetWorkflow : datasetWorkflows) {
            var cloudlet = createCloudlets(brokerId, datasetWorkflow);
            cloudlets.add(cloudlet);
        }
        return cloudlets;
    }
}
