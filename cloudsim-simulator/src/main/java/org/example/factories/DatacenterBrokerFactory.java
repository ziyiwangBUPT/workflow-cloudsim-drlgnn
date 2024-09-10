package org.example.factories;

import lombok.Builder;
import org.example.entities.MonitoredDatacenterBroker;

import java.util.concurrent.atomic.AtomicInteger;

@Builder
public class DatacenterBrokerFactory {
    private static final AtomicInteger CURRENT_DC_ID = new AtomicInteger(0);

    public MonitoredDatacenterBroker createBroker() {
        try {
            var brokerName = String.format("Broker-%d", CURRENT_DC_ID.getAndIncrement());
            return new MonitoredDatacenterBroker(brokerName);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
