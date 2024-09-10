package org.example.tables;

import org.cloudbus.cloudsim.Host;
import org.example.entities.MonitoredHost;

import java.util.Collection;
import java.util.List;

public class HostTable extends AbstractTable<MonitoredHost> {
    private static final int K = 1024;

    public HostTable(Collection<? extends MonitoredHost> list) {
        super(list);
    }

    @Override
    protected void createTableColumns() {
        addColumn("DC", ID_UNIT, STRING_FORMAT, host -> host.getDatacenter().getId());
        addColumn("Host", ID_UNIT, STRING_FORMAT, Host::getId);

        addColumn("PES", COUNT_UNIT, INTEGER_FORMAT, Host::getNumberOfPes);
        addColumn("Speed ", GIPS_UNIT, DECIMAL_FORMAT, host -> host.getTotalMips() / K);
        addColumn("Ram", GB_UNIT, INTEGER_FORMAT, host -> host.getRam() / K);
        addColumn(" BW  ", GB_S_UNIT, DECIMAL_FORMAT, host -> (double) host.getBw() / K);
        addColumn("Storage", GB_UNIT, INTEGER_FORMAT, host -> host.getStorage() / 1000);

        addColumn("VMs", COUNT_UNIT, INTEGER_FORMAT, host -> host.getGuestList().size());
        addColumn("CPU Usage", PERC_UNIT, DECIMAL_FORMAT, host -> host.getAverageCpuUtilization() * 100);
    }
}
