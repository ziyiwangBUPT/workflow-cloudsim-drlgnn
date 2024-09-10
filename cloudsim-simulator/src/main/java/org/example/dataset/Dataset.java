package org.example.dataset;

import com.google.gson.FieldNamingPolicy;
import com.google.gson.GsonBuilder;
import lombok.Data;

import java.util.List;

@Data
public class Dataset {
    private final List<DatasetWorkflow> workflows;
    private final List<DatasetVm> vms;
    private final List<DatasetHost> hosts;

    /// Convert a JSON string to a Dataset object.
    public static Dataset fromJson(String json) {
        var gson = new GsonBuilder().setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES).create();
        return gson.fromJson(json, Dataset.class);
    }
}
