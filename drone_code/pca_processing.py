import cv2
import numpy as np
import struct
from lora_send import send_lora_payload

pca_components = np.load("pca_components.npy")
pca_mean = np.load("pca_mean.npy")

img = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (32, 32))
flat = img.flatten() / 255.0

centered = flat - pca_mean
pca_vector = np.dot(pca_components, centered)

scaled = np.clip((pca_vector * 100).astype(np.int16), -128, 127).astype(np.int8)
payload = scaled.tobytes()

print("LoRa payload bytes:", payload)
send_lora_payload(payload)