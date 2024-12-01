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
MAX_TRAINING_DS_SEED = 1_000_000_000
MIN_TESTING_DS_SEED = MAX_TRAINING_DS_SEED + 1
MIN_EVALUATING_DS_SEED = 2 * MAX_TRAINING_DS_SEED + 1

# Evaluation related
# ----------------------------------------------------------------------------------------------------------------------

DEFAULT_MODEL_DIR = ROOT_PATH / "logs"
ALGORITHMS = [
    # ("CP-SAT", "cp_sat"),
    ("HEFT", "insertion_heft"),
    # ("GA", "ga"),
    ("GIN - 4x10x20 - 32H (1/speed)", "gin:1732731101_gin_inv_speed:model_645120.pt"),
    # ("GIN 4x200 1024 STEPS", "gin:1732909949_gin_mke_r1024:model_399360.pt"),
    # ("GIN 4x200 STEPPED", "gin:1733026564_gin_mk_tinc:model_102400.pt"),
]
