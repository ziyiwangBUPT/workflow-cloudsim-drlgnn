package org.example.tables;

import lombok.NonNull;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.Function;

/// Class that represents a table with columns and rows.
/// Inspired from CloudSim Plus table classes.
public abstract class AbstractTable<T> {
    protected static final String STRING_FORMAT = "%s";
    protected static final String DECIMAL_FORMAT = "%.2f";
    protected static final String INTEGER_FORMAT = "%d";

    protected static final String ID_UNIT = "ID";
    protected static final String COUNT_UNIT = "#";
    protected static final String GIPS_UNIT = "GIPS";
    protected static final String GB_UNIT = "GB";
    protected static final String GB_S_UNIT = "GB/s";
    protected static final String MI_UNIT = "MI";
    protected static final String S_UNIT = "s";
    protected static final String PERC_UNIT = "%%";

    private final List<TableColumn<T>> columns = new ArrayList<>();
    private final Collection<? extends T> rows;

    protected AbstractTable(@NonNull Collection<? extends T> rows) {
        this.rows = rows;
    }

    protected abstract void createTableColumns();

    protected void addColumn(@NonNull String title, @NonNull String subtitle,
                             @NonNull String format, @NonNull Function<T, Object> dataFunction) {
        columns.add(new TableColumn<>(title, subtitle, format, dataFunction));
    }

    public void print() {
        createTableColumns();
        System.out.println();
        System.out.println(separatedData(TableColumn::title));
        System.out.println(separatedData(TableColumn::subtitle));
        System.out.println(separatedData(c -> "-".repeat(c.columnWidth())));
        for (T row : rows) {
            System.out.println(separatedData(c -> String.format(c.format(), c.dataFunction().apply(row))));
        }
        System.out.println();
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
