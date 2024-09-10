# Dataset Geneterator

This is a simple dataset generator that creates a dataset with a Directed Acyclic Graph (DAG) structured workflows and VMs.
This project uses UV (https://github.com/astral-sh/uv) to manage the dependencies.

## Installation

```bash
$ uv sync
```

## Check Types

```bash
$ uv run mypy src
```

## Scripts

### Generate Dataset

```bash
$ python src/dataset_visualizer/gen_dataset.py --help
Usage: gen_dataset.py [OPTIONS]

Options:
  --seed INTEGER                Random seed
  --host_count INTEGER          Number of hosts
  --vm_count INTEGER            Number of VMs
  --max_cores INTEGER           Maximum number of cores per VM
  --min_cpu_speed_mips INTEGER  Minimum CPU speed in MIPS
  --max_cpu_speed_mips INTEGER  Maximum CPU speed in MIPS
  --workflow_count INTEGER      Number of workflows
  --min_task_count INTEGER      Minimum number of tasks per workflow
  --max_task_count INTEGER      Maximum number of tasks per workflow
  --min_task_length INTEGER     Minimum task length
  --max_task_length INTEGER     Maximum task length
  --arrival_rate INTEGER        Arrival rate of workflows
  --help                        Show this message and exit.

# Usage
$ python src/dataset_visualizer/gen_dataset.py > tmp/dataset.json
```

### Solve Dataset

This will solve using the CP-SAT solver (or round robin) and generate some charts in a tmp directory in current directory. (directory must exist)

- `workflows.png` - Shows the workflow graphs.
- `execution.png` - Shows the execution graph.
- `gantt.png` - Shows the Gantt chart.

```bash
$ python src/dataset_visualizer/solve_dataset.py --help
Usage: solve_dataset.py [OPTIONS]

Options:
  --method [sat|round_robin]  Method to solve the dataset
  --help                      Show this message and exit.

# Usage
$ python src/dataset_visualizer/solve_dataset.py < tmp/dataset.json
```

## Notes

Hosts specs used to generate the datasets can be found in `data/host_specs.json`. Add more specs to generate more diverse datasets.
