# File: basestation_code/rf_training.py
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Get script directory
BASE_DIR = os.path.dirname(__file__)

# Load PCA-transformed features
X_pca_path = os.path.join(BASE_DIR, "X_pca.npy")
X_pca = np.load(X_pca_path)

# Create matching labels
labels = []
for folder in ["no_person", "person"]:
    label = 0 if folder == "no_person" else 1
    folder_path = os.path.join(BASE_DIR, "data", folder)
    n = len(os.listdir(folder_path))
    labels += [label] * n

y = np.array(labels)

# Train Random Forest
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Evaluate and save
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

model_path = os.path.join(BASE_DIR, "rf_model.pkl")
joblib.dump(clf, model_path)
print(f"✅ RF model saved to {model_path}")
