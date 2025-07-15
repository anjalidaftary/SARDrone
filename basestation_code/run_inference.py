import joblib
import numpy as np
import os
from sklearn.preprocessing import normalize
# from lora_receive import receive_lora_payload  # comment out when emulating LoRa

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")

clf = joblib.load(MODEL_PATH)

# Simulate receiving LoRa payload from file
# payload = receive_lora_payload(timeout=10.0)   # ←comment out when emulating LoRa
try:
    with open("pca_payload.bin", "rb") as f:
        payload = f.read()
except FileNotFoundError:
    print("pca_payload.bin not found.")
    exit()

# comment out when emulating LoRa
# if payload is None:
#     print("No data received. Exiting.")
#     exit()

# Convert bytes back to PCA vector
vector = np.frombuffer(payload, dtype=np.int8).astype(np.float32) / 100.0
pca_vector = vector.reshape(1, -1)
# pca_vector = normalize(pca_vector)

# Run inference
label = clf.predict(pca_vector)[0]
prob = clf.predict_proba(pca_vector)[0][1]

if prob > 0.75:
    print(f"🚨 Person Detected! Confidence: {prob:.2f}")
else:
    print(f"No Person Detected. Confidence: {1 - prob:.2f}")

