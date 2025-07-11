import zipfile
import os

ZIP_PATH = "basestation_code/annotations.zip"
EXTRACT_TO = "basestation_code"
TARGET_FILE = os.path.join(EXTRACT_TO, "instances_train2017.json")

print("Extracting COCO annotation JSON...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_TO)

print("Extraction complete. You can now run your dataset script.")