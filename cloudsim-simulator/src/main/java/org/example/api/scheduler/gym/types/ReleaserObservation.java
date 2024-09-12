package org.example.api.scheduler.gym.types;


public record ReleaserObservation(int cachedJobs, int vmCount, double completionTimeVariance) {
}
