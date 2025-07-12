import subprocess
import os
import sys

VENV_PYTHON = sys.executable  # Dynamically get the path to the current interpreter

def run_script(script_name):
    result = subprocess.run([VENV_PYTHON, script_name], cwd="basestation_code")
    if result.returncode != 0:
        print(f"Error in {script_name}, aborting.")
        sys.exit(1)

print("\n🔧 Starting Full Basestation Pipeline Setup\n")

# 1. Unzip annotations if needed
if not os.path.exists("basestation_code/instances_train2017.json"):
    print("[1] Unzipping annotations...")
    run_script("unzip_annotations.py")
else:
    print("[1] Annotations already unzipped.")

# 2. Load COCO data and download images
print("[2] Downloading COCO dataset images...")
run_script("load_COCO_data.py")

# 3. Train PCA
print("[3] Training PCA...")
run_script("pca_training.py")

# 4. Train Random Forest
print("[4] Training Random Forest...")
run_script("rf_training.py")

'''
5. Run inference with LoRa
print("[5] Running inference...")
run_script("run_inference.py")
'''

print("\nFull pipeline executed ")