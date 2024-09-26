package org.example.sensors;

import lombok.Getter;

/// A sensor to keep track of the state of tasks in the system.
@Getter
public class TaskStateSensor {
    @Getter
    private static final TaskStateSensor instance = new TaskStateSensor();

    private int bufferedTasks = 0; // Just submitted by user (Buffer)
    private int scheduledTasks = 0; // Scheduled to a VM to run (Scheduler)
    private int executedTasks = 0; // Actually running in VM (Executor)
    private int completedTasks = 0; // Finished running (Executor)

    private TaskStateSensor() {
    }

    /// Increment the number of tasks in the buffer.
    public void bufferTasks(int count) {
        bufferedTasks += count;
    }

    /// Increment the number of tasks scheduled to run.
    public void scheduleTasks(int count) {
        scheduledTasks += count;
    }

    /// Increment the number of tasks actually running.
    public void executeTasks(int count) {
        executedTasks += count;
    }

    /// Increment the number of tasks that have completed.
    public void completeTasks(int count) {
        completedTasks += count;
    }

    /// Get the number of tasks that are currently running/not completed.
    public int getIncompleteTasks() {
        return bufferedTasks - completedTasks;
    }
}
