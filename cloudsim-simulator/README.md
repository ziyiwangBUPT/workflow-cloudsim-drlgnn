# CloudSim Simulator for Dynamic Workflow Scheduling

This project is a project depending on CloudSim project adding following capabilities.

## Building

First install the CloudSim jar into the local maven repository.

```bash
$ mvn install:install-file -Dfile=cloudsim/cloudsim-7.0.0-alpha.jar -DgroupId=org.cloudbus -DartifactId=cloudsim -Dversion=7.0.0-alpha -Dpackaging=jar
```

Then you can build the project by running the following command.

```bash
$ mvn clean package
```

## Running

```bash
$  java -jar <jar-file> --help -f=<dataset-file>
Usage: CloudSim Simulator [-hV] [-d=<simulationDuration>] -f=<datasetFile>
Runs a simulation of a workflow scheduling algorithm.
  -d, --duration=<simulationDuration>
                             Duration of the simulation
  -f, --file=<datasetFile>   Dataset file
  -h, --help                 Show this help message and exit.
  -V, --version              Print version information and exit.
```

You can generate the jar file by running the following command.

```bash
$ maven clean package
```

You can generate the dataset by using dataset generator project.
Please refer to the [dataset generator project](../dataset-generator/README.md) for more information.

## Dataset

The dataset file should be in the following format.

```json5
{
  "workflows": [
    {
      "id": 0,
      "arrival_time": 0,
      "tasks": [
        {
          "id": 0,
          "workflow_id": 0,
          "length": 58487,
          "req_cores": 1,
          "child_ids": [
            1,
            2
          ]
        },
        {
          "id": 1,
          "workflow_id": 0,
          "length": 47957,
          "req_cores": 5,
          "child_ids": []
        },
        // ...
      ]
    }
    // ...
  ],
  "vms": [
    {
      "id": 0,
      "host_id": 0,
      "cores": 5,
      "cpu_speed_mips": 2506,
      "memory_mb": 512,
      "disk_mb": 1024,
      "bandwidth_mbps": 50,
      "vmm": "Xen"
    },
    // ...
  ],
  "hosts": [
    {
      "id": 0,
      "cores": 56,
      "cpu_speed_mips": 604800,
      "memory_mb": 65536,
      "disk_mb": 10000000,
      "bandwidth_mbps": 1536,
      "power_idle_watt": 50,
      "power_peak_watt": 432
    }
    // ...
  ]
}
```