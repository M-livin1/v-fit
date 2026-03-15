import os
import subprocess
import sys

print(" Cloth pipeline started")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
API_DIR = os.path.join(PROJECT_ROOT, "src", "api")

def run(script_name):
    script_path = os.path.join(API_DIR, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f" {script_name} not found")

    subprocess.run(
        [sys.executable, script_path],
        check=True
    )

run("preprocessing.py")
run("pose.py")
run("overlay.py")

print(" Cloth pipeline finished successfully")
