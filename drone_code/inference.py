from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import os

interpreter = tflite.Interpreter(model_path="yolov5n.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((640, 640))
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_np, 0), img, img_resized.size

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_iou(box1, box2):
    xa = max(box1[0], box2[0]); ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2]); yb = min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-6)

def postprocess(output_data, original_image, conf_thresh=0.5):
    w, h = original_image.size
    crops = []
    predictions = output_data[0]

    for det in predictions:
        if len(det) < 6:
            continue  # skip invalid rows

        x_center, y_center, box_w, box_h = det[0:4]
        obj_conf = det[4]
        class_probs = det[5:]
        class_id = np.argmax(class_probs)
        class_score = class_probs[class_id]

        conf = obj_conf * class_score

        if conf > conf_thresh and class_id == 0:  # class 0 = person
            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = original_image.crop((x1, y1, x2, y2))
            crops.append(crop)

    return crops
