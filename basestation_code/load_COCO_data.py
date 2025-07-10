import os
import requests
from pycocotools.coco import COCO
from PIL import Image
from io import BytesIO

PERSON_COUNT = 150
NOPERSON_COUNT = 150

os.makedirs("data/person", exist_ok=True)
os.makedirs("data/no_person", exist_ok=True)

print("Loading COCO annotations")
coco = COCO("instances_train2017.json")

# Loads images with people
person_cat_id = coco.getCatIds(catNms=["person"])[0]
person_img_ids = coco.getImgIds(catIds=[person_cat_id])

print("Downloading images with a person")
for idx, img_id in enumerate(person_img_ids[:PERSON_COUNT]):
    info = coco.loadImgs(img_id)[0]
    url = info['coco_url']
    try:
        resp = requests.get(url)
        img = Image.open(BytesIO(resp.content))
        img.save(f"data/person/{idx:03d}.jpg")
        print(f"Saved person/{idx:03d}.jpg")
    except:
        print(f"Failed to download {url}")

# For image IDs with no people
all_ids = coco.getImgIds()
no_person_ids = [i for i in all_ids if not coco.getAnnIds(imgIds=[i], catIds=[person_cat_id])]

print("Downloading no_pe rson images")
for idx, img_id in enumerate(no_person_ids[:NOPERSON_COUNT]):
    info = coco.loadImgs(img_id)[0]
    url = info['coco_url']
    try:
        resp = requests.get(url)
        img = Image.open(BytesIO(resp.content))
        img.save(f"data/no_person/{idx:03d}.jpg")
        print(f"Saved no_person/{idx:03d}.jpg")
    except:
        print(f"Failed to download {url}")

print("Dataset ready in data folder")
