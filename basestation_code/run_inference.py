import joblib
import numpy as np
from lora_receive import receive_lora_payload

clf = joblib.load("rf_model.pkl")

# Receive LoRa payload (blocking with timeout)
payload = receive_lora_payload(timeout=10.0)

if payload is None:
    print("No data received. Exiting.")
    exit()

# Convert bytes back to PCA vector
vector = np.frombuffer(payload, dtype=np.int8).astype(np.float32) / 100.0
pca_vector = vector.reshape(1, -1)

# Run inference
label = clf.predict(pca_vector)[0]
prob = clf.predict_proba(pca_vector)[0][1]

if label == 1:
    print(f"Person Detected! Confidence: {prob:.2f}")
else:
    print(f"No Person Detected. Confidence: {1 - prob:.2f}")
