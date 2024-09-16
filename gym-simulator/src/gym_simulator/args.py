from dataclasses import dataclass


@dataclass
class TrainArgs:
    simulator: str
    """Path to the simulator JAR file"""

    dataset: str
    """Path to the dataset JSON file"""

    render_mode: str | None = None
    """Render mode"""


@dataclass
class TestArgs:
    checkpoint_dir: str
    """Path to the checkpoint directory"""

    simulator: str
    """Path to the simulator JAR file"""

    dataset: str
    """Path to the dataset JSON file"""

    render_mode: str | None = None
    """Render mode"""
