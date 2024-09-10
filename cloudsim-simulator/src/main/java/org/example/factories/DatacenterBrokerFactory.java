package org.example.factories;

import lombok.Builder;
import org.example.entities.DynamicDatacenterBroker;

import java.util.concurrent.atomic.AtomicInteger;

/// Factory for creating Datacenter brokers.
@Builder
public class DatacenterBrokerFactory {
    private static final AtomicInteger CURRENT_DC_ID = new AtomicInteger(0);

    public DynamicDatacenterBroker createBroker() {
        try {
            var brokerName = String.format("Broker-%d", CURRENT_DC_ID.getAndIncrement());
            return new DynamicDatacenterBroker(brokerName);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
