package org.example.dataset;

import lombok.Data;
import org.example.utils.GsonHelper;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Scanner;

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

    public static Dataset fromFile(File file) {
        try (var reader = new FileReader(file);
             var scanner = new Scanner(reader)
        ) {
            var json = scanner.useDelimiter("\\A").next();
            return fromJson(json);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}