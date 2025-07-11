# File: basestation_code/train_pca.py
import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
import sys

# Determine script’s directory, then data folder
BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, "data")

# Collect all image vectors
X = []
for folder in ["person", "no_person"]:
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        print(f"ERROR: Data folder not found: {folder_path}")
        sys.exit(1)

    files = os.listdir(folder_path)
    if not files:
        print(f"ERROR: No images in {folder_path}")
        sys.exit(1)

    for filename in files:
        path = os.path.join(folder_path, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping non-image file: {path}")
            continue
        img = cv2.resize(img, (32, 32))
        X.append(img.flatten() / 255.0)

X = np.array(X)
print(f"Loaded {X.shape[0]} images, each flattened to {X.shape[1] if X.ndim>1 else 0} features")

if X.ndim != 2 or X.shape[0] < 2:
    print("ERROR: Need at least 2 images to train PCA")
    sys.exit(1)

pca = PCA(n_components=20)
X_pca = pca.fit_transform(X)

np.save(os.path.join(BASE_DIR, "pca_components.npy"), pca.components_)
np.save(os.path.join(BASE_DIR, "pca_mean.npy"), pca.mean_) # mean vector for centering images
np.save(os.path.join(BASE_DIR, "X_pca.npy"), X_pca) # post-processed training set
print("PCA training complete and files saved.")
