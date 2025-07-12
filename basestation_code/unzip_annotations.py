import zipfile
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "annotations.zip")
TARGET_NAME = "instances_train2017.json"
TARGET_PATH = os.path.join(BASE_DIR, TARGET_NAME)

print("Extracting COCO annotation JSON...")

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    for member in zip_ref.namelist():
        if member.endswith(TARGET_NAME):
            with zip_ref.open(member) as source, open(TARGET_PATH, "wb") as target:
                target.write(source.read())
            print(f"Extracted {TARGET_NAME} to {TARGET_PATH}")
            break
    else:
        print(f"{TARGET_NAME} not found in ZIP archive.")
