package org.example.dataset;

import com.google.gson.FieldNamingPolicy;
import com.google.gson.GsonBuilder;
import lombok.Data;

import java.util.List;

@Data
public class DatasetSolution {
    private final Dataset dataset;
    private final List<DatasetExecution> executions;

    /// Convert to a JSON string.
    public String toJson() {
        var gson = new GsonBuilder().setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES).create();
        return gson.toJson(this);
    }
}
