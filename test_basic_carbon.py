import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing carbon intensity...")

from scheduler.config.carbon_intensity import CARBON_INTENSITY_DATA, FIXED_NUM_HOSTS

print(f"Hosts: {FIXED_NUM_HOSTS}")
print(f"Carbon data shape: {len(CARBON_INTENSITY_DATA)} x {len(CARBON_INTENSITY_DATA[0])}")
print("SUCCESS!")

