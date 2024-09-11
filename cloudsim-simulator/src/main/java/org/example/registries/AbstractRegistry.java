package org.example.registries;

import org.example.utils.SummaryTable;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

public abstract class AbstractRegistry<T> {
    private final Map<Integer, T> map = new HashMap<>();

    public T findRegistered(int id) {
        return map.get(id);
    }

    public int getSize() {
        return map.size();
    }

    public void printSummaryTable() {
        var table = buildSummaryTable();
        table.addRows(map.values());
        table.print();
    }

    protected Stream<T> itemStream() {
        return map.values().stream();
    }

    protected void register(int id, T item) {
        map.put(id, item);
    }

    protected abstract SummaryTable<T> buildSummaryTable();
}
