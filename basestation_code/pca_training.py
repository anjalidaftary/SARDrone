from sklearn.decomposition import PCA
import numpy as np
import cv2
import os

X = []

for folder in ["person", "no_person"]:
    for filename in os.listdir(f"data/{folder}"):
        img = cv2.imread(f"data/{folder}/{filename}", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32)) # experiment with diff values based on pi capacity
        flat = img.flatten() / 255.0
        X.append(flat)

X = np.array(X)

pca = PCA(n_components=20) # ****TEST LATER IF CONFIDENCE LOW****
X_pca = pca.fit_transform(X)

np.save("pca_components.npy", pca.components_)
np.save("pca_mean.npy", pca.mean_)
np.save("X_pca.npy", X_pca)