import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path("/Users/shweta/Desktop/vit chennai/major project fy/MediGraph")

def make_bundle():
    script_path = BASE_DIR / "experiments/create_audit_bundle.py"
    print(f"Executing {script_path}...")
    res = subprocess.run([sys.executable, str(script_path)], check=True)
    print("Bundle v2 updated successfully.")

if __name__ == "__main__":
    make_bundle()

