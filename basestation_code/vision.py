import subprocess
import json
import os

image = 'test_image.jpg'
output_image = 'output_overhead.png'
prediction_json = 'your_image.jpg.predictions.json'

# Run OpenPifPaf via CLI
subprocess.run([
    'python', '-m', 'openpifpaf.predict',
    image,
    '--image-output', output_image
])

# Count detections from the predictions JSON
if os.path.exists(prediction_json):
    with open(prediction_json, 'r') as f:
        predictions = json.load(f)
        num_people = len(predictions)
        print(f"✅ Detected {num_people} humans")
else:
    print("⚠️ No prediction file found. Make sure OpenPifPaf finished successfully.")

print(f"✅ Saved overlaid output to {output_image}")
