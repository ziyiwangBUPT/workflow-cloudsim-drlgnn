package org.example.api.scheduler.gym.types;

import lombok.Data;


@Data
public final class AgentResult<TObservation> {
    private final TObservation observation;
    private final double reward;
    private final boolean terminated;
    private final boolean truncated;

    public static <TObservation> AgentResult<TObservation> reward(TObservation observation, double reward) {
        return new AgentResult<>(observation, reward, false, false);
    }

    public static <TObservation> AgentResult<TObservation> truncated(double reward) {
        return new AgentResult<>(null, reward, false, true);
    }
}
