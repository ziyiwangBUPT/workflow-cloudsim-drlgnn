package org.example.ticks;

import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.CloudSimTags;
import org.cloudbus.cloudsim.core.SimEntity;
import org.cloudbus.cloudsim.core.SimEvent;

public abstract class SimulationTickListener extends SimEntity {
    private static final int TICK_EMIT_PERIOD_S = 1;

    protected SimulationTickListener(String name) {
        super(name);
    }

    @Override
    public void processEvent(SimEvent ev) {
        if (ev.getTag() == Tags.TICK) {
            CloudSim.pauseSimulation();
            onTick(ev.eventTime());
            schedule(getId(), TICK_EMIT_PERIOD_S, Tags.TICK);
            CloudSim.resumeSimulation();
        } else {
            Log.println("Unknown event received by " + getName() + ". Tag: " + ev.getTag());
        }
    }

    @Override
    public void startEntity() {
        super.startEntity();
        schedule(getId(), TICK_EMIT_PERIOD_S, Tags.TICK);
    }

    protected abstract void onTick(double timeMs);

    protected enum Tags implements CloudSimTags {
        TICK
    }
}
