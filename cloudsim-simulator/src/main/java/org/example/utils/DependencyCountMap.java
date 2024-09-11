package org.example.utils;

import java.util.HashMap;
import java.util.Map;

public class DependencyCountMap<T> {
    private final Map<T, Integer> countMap = new HashMap<>();

    public void addNewDependency(T key) {
        countMap.put(key, countMap.getOrDefault(key, 0) + 1);
    }

    public void removeOneDependency(T key) {
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

    public boolean hasNoDependency(T key) {
        return !countMap.containsKey(key);
    }

    @Override
    public String toString() {
        return countMap.toString();
    }
}
