package org.example.core.registries;

import org.example.utils.SummaryTable;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

/// Abstract class for a registry of items.
public abstract class AbstractRegistry<T> {
    private final Map<Integer, T> map = new HashMap<>();

    /// Find an item by its ID.
    public T findRegistered(int id) {
        return map.get(id);
    }

    /// Get the number of items in the registry.
    public int getSize() {
        return map.size();
    }

    /// Print a summary table of all items in the registry.
    public void printSummaryTable() {
        var table = buildSummaryTable();
        table.addRows(map.values());
        table.print();
    }

    /// Get a stream of all items in the registry.
    protected Stream<T> itemStream() {
        return map.values().stream();
    }

    /// Register an item in the registry.
    protected void register(int id, T item) {
        map.put(id, item);
    }

    /// Build a summary table of all items in the registry.
    protected abstract SummaryTable<T> buildSummaryTable();
}
