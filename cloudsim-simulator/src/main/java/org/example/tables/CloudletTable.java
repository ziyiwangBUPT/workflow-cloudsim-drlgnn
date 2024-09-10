package org.example.tables;

import org.cloudbus.cloudsim.Cloudlet;

import java.util.List;

public class CloudletTable extends AbstractTable<Cloudlet> {
    public CloudletTable(List<? extends Cloudlet> list) {
        super(list);
    }

    @Override
    protected void createTableColumns() {
        addColumn("Cloudlet", ID_UNIT, STRING_FORMAT, Cloudlet::getCloudletId);
        addColumn("Guest", ID_UNIT, STRING_FORMAT, Cloudlet::getGuestId);

        addColumn("    Status    ", ID_UNIT, STRING_FORMAT, cloudlet -> cloudlet.getStatus().name());
        addColumn("Cloudlet Len", MI_UNIT, INTEGER_FORMAT, Cloudlet::getCloudletLength);
        addColumn("Finished Len", MI_UNIT, INTEGER_FORMAT, Cloudlet::getCloudletFinishedSoFar);
        addColumn("Start Time", S_UNIT, DECIMAL_FORMAT, Cloudlet::getExecStartTime);
        addColumn("End Time  ", S_UNIT, DECIMAL_FORMAT, Cloudlet::getExecFinishTime);
        addColumn("Makespan  ", S_UNIT, DECIMAL_FORMAT, cloudlet -> cloudlet.isFinished() ? cloudlet.getExecFinishTime() - cloudlet.getExecStartTime() : 0);
    }
}
