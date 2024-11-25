package org.example.dataset;

import lombok.Data;
import org.example.utils.GsonHelper;

import java.util.List;

@Data
public class DatasetSolution {
    private final Dataset dataset;
    private final List<DatasetVmAssignment> vmAssignments;

    /// Convert to a JSON string.
    public String toJson() {
        var gson = GsonHelper.getGson();
        return gson.toJson(this);
    }
}
