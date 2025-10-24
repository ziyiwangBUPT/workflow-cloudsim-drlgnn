from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent

# Data files
# ----------------------------------------------------------------------------------------------------------------------

DATA_PATH = ROOT_PATH / "data"
HOST_SPECS_PATH = DATA_PATH / "host_specs.json"
WORKFLOW_FILES = [
    DATA_PATH / "pegasus_workflows" / "example.dag",      # Diamond 工作流（简单）
    DATA_PATH / "pegasus_workflows" / "montage.dag",      # Montage 工作流（天文图像处理）
    DATA_PATH / "pegasus_workflows" / "genome.dag",       # Genome 工作流（基因组学）
    DATA_PATH / "pegasus_workflows" / "sipht.dag",        # SIPHT 工作流（生物信息学）
    DATA_PATH / "pegasus_workflows" / "cybershake.dag",   # CyberShake 工作流（地震模拟）
    DATA_PATH / "pegasus_workflows" / "inspiral.dag",     # Inspiral 工作流（引力波分析）
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
    ("Random", "random"),
    ("Round Robin", "round_robin"),
    ("Min-Min", "min_min"),
    ("Max-Min", "max_min"),
    ("HEFT", "insertion_heft"),
    ("Proposed", "gin:1733128027_gin_R=-diff:model_501760.pt"),
]
