package org.example.utils;

import lombok.NonNull;

import java.util.HashMap;
import java.util.Map;

/// A map that keeps track of the number of dependencies for each key.
public class DependencyCountMap<T> {
    private final Map<T, Integer> countMap = new HashMap<>();

    /// Adds a new dependency for the given key.
    public void addNewDependency(@NonNull T key) {
        countMap.put(key, countMap.getOrDefault(key, 0) + 1);
    }

    /// Removes one dependency for the given key.
    public void removeOneDependency(@NonNull T key) {
        var currentCount = countMap.get(key);
        if (currentCount == null) {
            throw new IllegalArgumentException("The key does not have any dependency: " + key);
        }

        if (currentCount == 1) {
            countMap.remove(key);
        } else {
            countMap.put(key, currentCount - 1);
        }
    }

    /// Returns true if the given key has no dependency.
    public boolean hasNoDependency(@NonNull T key) {
        return !countMap.containsKey(key);
    }

    @Override
    public String toString() {
        return countMap.toString();
    }
}
