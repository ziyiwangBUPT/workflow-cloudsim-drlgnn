# Gym Simlator

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
$ uv run src/dataset_generator/gen_dataset.py --help
usage: gen_dataset.py [-h] [OPTIONS]

╭─ options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                                                         │
│ --seed INT              random seed (default: 42)                                                               │
│ --host-count INT        number of hosts (default: 2)                                                            │
│ --vm-count INT          number of VMs (default: 4)                                                              │
│ --max-memory-gb INT     maximum amount of RAM for a VM (in GB) (default: 10)                                    │
│ --min-cpu-speed INT     minimum CPU speed in MIPS (default: 500)                                                │
│ --max-cpu-speed INT     maximum CPU speed in MIPS (default: 5000)                                               │
│ --workflow-count INT    number of workflows (default: 3)                                                        │
│ --dag-method STR        DAG generation method (pegasus, gnp) (default: gnp)                                     │
│ --gnp-min-n INT         minimum number of tasks per workflow (for G(n,p) method) (default: 1)                   │
│ --gnp-max-n INT         maximum number of tasks per workflow (for G(n,p) method) (default: 5)                   │
│ --task-length-dist STR  task length distribution (normal, uniform, left_skewed, right_skewed) (default: normal) │
│ --min-task-length INT   minimum task length (default: 500)                                                      │
│ --max-task-length INT   maximum task length (default: 100000)                                                   │
│ --task-arrival STR      task arrival mode (static, dynamic) (default: dynamic)                                  │
│ --arrival-rate FLOAT    arrival rate of workflows/second (for dynamic arrival) (default: 3)                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

# Usage
$ uv run src/dataset_generator/gen_dataset.py > logs/data/dataset.json
```

### Solve Dataset

This will solve using the CP-SAT solver (or round robin) and generate the execution solution.

```bash
$ uv run src/dataset_generator/solve_dataset.py --help
usage: solve_dataset.py [-h] [--method STR]

╭─ options ─────────────────────────────────────────────────────────────────────────╮
│ -h, --help          show this help message and exit                               │
│ --method STR        method to solve the dataset (sat, round_robin) (default: sat) │
╰───────────────────────────────────────────────────────────────────────────────────╯

# Usage
$ uv run src/dataset_generator/solve_dataset.py < logs/data/dataset.json > logs/data/solution.json
```

### Solve Dataset

This will generate some charts current directory with a prefix given. (directories in prefix must exist)

- `PREFIX_workflows.png` - Shows the workflow graphs.
- `PREFIX_execution.png` - Shows the execution graph.
- `PREFIX_gantt.png` - Shows the Gantt chart.

```bash
$ uv run src/dataset_generator/viz_solution.py --help
usage: viz_solution.py [-h] [--prefix STR]

╭─ options ───────────────────────────────────────────────────────────────────────────╮
│ -h, --help          show this help message and exit                                 │
│ --prefix STR        file prefix to use (with directory) (default: tmp/viz_solution) │
╰─────────────────────────────────────────────────────────────────────────────────────╯

# Usage
$ uv run src/dataset_generator/viz_solution.py --prefix logs/data/viz < logs/data/solution.json
```

## Notes

Hosts specs used to generate the datasets can be found in `data/host_specs.json`. Add more specs to generate more diverse datasets.
