package org.example.schedulers;

import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.GuestEntity;
import org.example.entities.ExecutionPlan;
import org.example.entities.WorkflowCloudlet;

import java.util.List;

public interface Scheduler {
    ExecutionPlan schedule(List<List<WorkflowCloudlet>> workflows, List<Vm> vms);
}
