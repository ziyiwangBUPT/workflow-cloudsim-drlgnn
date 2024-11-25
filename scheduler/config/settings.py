from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent

# Data files
# ----------------------------------------------------------------------------------------------------------------------

DATA_PATH = ROOT_PATH / "data"
HOST_SPECS_PATH = DATA_PATH / "host_specs.json"
WORKFLOW_FILES = [
    DATA_PATH / "pegasus_workflows" / "sipht.dag",
]


# Model related
# ----------------------------------------------------------------------------------------------------------------------

MAX_OBS_SIZE = 100_000


# Evaluation related
# ----------------------------------------------------------------------------------------------------------------------

ALGORITHMS = [
    ("CP-SAT", "cp_sat"),
    ("Round Robin", "round_robin"),
    ("Min-Min", "min_min"),
    ("Best Fit", "best_fit"),
    ("Max-Min", "max_min"),
    ("HEFT", "heft"),
]
