import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load PCA-transformed features
X_pca = np.load("X_pca.npy")

labels = []
for folder in ["no_person", "person"]:
    label = 0 if folder == "no_person" else 1
    n = len(os.listdir(f"data/{folder}"))
    labels += [label] * n

y = np.array(labels)

# Train Random Forest
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
joblib.dump(clf, "rf_model.pkl")