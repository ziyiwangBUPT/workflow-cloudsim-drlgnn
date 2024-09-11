package org.example.utils;

import lombok.NonNull;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.Function;

/// Class that represents a table with columns and rows.
/// Inspired from CloudSim Plus table classes.
public class SummaryTable<T> {
    public static final String STRING_FORMAT = "%s";
    public static final String DECIMAL_FORMAT = "%.2f";
    public static final String INTEGER_FORMAT = "%d";

    public static final String ID_UNIT = "ID";
    public static final String COUNT_UNIT = "#";
    public static final String GIPS_UNIT = "GIPS";
    public static final String GB_UNIT = "GB";
    public static final String GB_S_UNIT = "GB/s";
    public static final String MI_UNIT = "MI";
    public static final String S_UNIT = "s";
    public static final String PERC_UNIT = "%%";

    private final List<TableColumn<T>> columns = new ArrayList<>();
    private final List<T> rows = new ArrayList<>();

    /// Add a new column to the table.
    public void addColumn(@NonNull String title, @NonNull String subtitle,
                          @NonNull String format, @NonNull Function<T, Object> dataFunction) {
        columns.add(new TableColumn<>(title, subtitle, format, dataFunction));
    }

    /// Add a new row to the table.
    public void addRows(@NonNull Collection<? extends T> newRows) {
        this.rows.addAll(newRows);
    }

    /// Print the table to the console.
    public void print() {
        System.err.println();
        System.err.println(separatedData(TableColumn::title));
        System.err.println(separatedData(TableColumn::subtitle));
        System.err.println(separatedData(c -> "-".repeat(c.columnWidth())));
        for (T row : rows) {
            System.err.println(separatedData(c -> String.format(c.format(), c.dataFunction().apply(row))));
        }
        System.err.println();
    }

    private String separatedData(Function<TableColumn<T>, String> mapper) {
        return "| " + columns.stream().map(rightAlignedMapper(mapper)).reduce((s1, s2) -> s1 + " | " + s2).orElse("") + " |";
    }

    private Function<TableColumn<T>, String> rightAlignedMapper(Function<TableColumn<T>, String> mapper) {
        return c -> String.format("%" + c.columnWidth() + "s", mapper.apply(c));
    }

    private record TableColumn<T>(String title, String subtitle, String format, Function<T, Object> dataFunction) {
        public int columnWidth() {
            return title.length();
        }
    }
}
