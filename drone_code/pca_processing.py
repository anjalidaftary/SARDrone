import cv2
import numpy as np
import struct
import os

# Load PCA components
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    pca_components = np.load(os.path.join(SCRIPT_DIR, "pca_components.npy"))
    pca_mean = np.load(os.path.join(SCRIPT_DIR, "pca_mean.npy"))

    print("✅ Loaded PCA parameters")
except FileNotFoundError:
    print("Missing PCA .npy files. Make sure you copied them to drone_code/")
    exit()

# Load input image
input_path = os.path.join(SCRIPT_DIR, "input.png")
img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("ERROR: input.png not found or unreadable!")
    exit()

img = cv2.resize(img, (32, 32))
flat = img.flatten() / 255.0

# Project with PCA
centered = flat - pca_mean
pca_vector = np.dot(pca_components, centered)

# Quantize
scaled = np.clip((pca_vector * 100).astype(np.int16), -128, 127).astype(np.int8)
payload = scaled.tobytes()

# send_lora_payload(payload) # comment out when emulating LoRA

# For emulating LoRa
with open("pca_payload.bin", "wb") as f:
    f.write(payload)
print("✅ pca_payload.bin saved in", os.getcwd())
