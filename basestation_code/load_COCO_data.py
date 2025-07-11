import shutil
import requests
from pycocotools.coco import COCO
from PIL import Image
from io import BytesIO
import os

# Determine this script’s folder
BASE_DIR = os.path.dirname(__file__)

# Paths to JSON and data folder
ANNOTATION_FILE = os.path.join(BASE_DIR, "instances_train2017.json")
DATA_DIR = os.path.join(BASE_DIR, "data")

PERSON_COUNT = 150
NOPERSON_COUNT = 150

# 1) Remove any existing data folder entirely
if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)

# 2) Recreate clean data/person and data/no_person directories
os.makedirs(os.path.join(DATA_DIR, "person"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "no_person"), exist_ok=True)

print("Loading COCO annotations from", ANNOTATION_FILE)
coco = COCO(ANNOTATION_FILE)

# 3) Download person images
person_cat_id = coco.getCatIds(catNms=["person"])[0]
person_img_ids = coco.getImgIds(catIds=[person_cat_id])

print("Downloading images with a person")
for idx, img_id in enumerate(person_img_ids[:PERSON_COUNT]):
    info = coco.loadImgs(img_id)[0]
    url = info['coco_url']
    try:
        resp = requests.get(url)
        img = Image.open(BytesIO(resp.content))
        img.save(os.path.join(DATA_DIR, "person", f"{idx:03d}.jpg"))
        print(f"Saved person/{idx:03d}.jpg")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

# 4) Download no_person images
all_ids = coco.getImgIds()
no_person_ids = [i for i in all_ids
                 if not coco.getAnnIds(imgIds=[i], catIds=[person_cat_id])]

print("Downloading no_person images")
for idx, img_id in enumerate(no_person_ids[:NOPERSON_COUNT]):
    info = coco.loadImgs(img_id)[0]
    url = info['coco_url']
    try:
        resp = requests.get(url)
        img = Image.open(BytesIO(resp.content))
        img.save(os.path.join(DATA_DIR, "no_person", f"{idx:03d}.jpg"))
        print(f"Saved no_person/{idx:03d}.jpg")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

print("Dataset ready in data/ folder")
