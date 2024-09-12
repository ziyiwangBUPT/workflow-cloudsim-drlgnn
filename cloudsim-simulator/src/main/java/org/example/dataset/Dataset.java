package org.example.dataset;

import lombok.Data;
import org.example.utils.GsonHelper;

import java.util.List;

@Data
public class Dataset {
    private final List<DatasetWorkflow> workflows;
    private final List<DatasetVm> vms;
    private final List<DatasetHost> hosts;

    /// Convert a JSON string to a Dataset object.
    public static Dataset fromJson(String json) {
        var gson = GsonHelper.getGson();
        return gson.fromJson(json, Dataset.class);
    }
}
