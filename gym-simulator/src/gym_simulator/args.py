from dataclasses import dataclass


@dataclass
class Args:
    simulator: str
    """Path to the simulator JAR file"""

    dataset: str
    """Path to the dataset JSON file"""

    render_mode: str | None = None
    """Render mode"""
