import os
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# 1) Load your TFLite model once
_MODEL_PATH = "yolov5n.tflite"
interpreter = tflite.Interpreter(model_path=_MODEL_PATH)
interpreter.allocate_tensors()
input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()

def preprocess_image(img_path, target_size=(640, 640)):
    """Load image, convert to RGB, resize, normalize, and add batch dim."""
    img = Image.open(img_path).convert("RGB")
    resized = img.resize(target_size)
    arr = np.array(resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0), img, target_size

def postprocess(output_data, original_image, resized_size, conf_thresh=0.5, iou_thresh=0.4):
    """
    Parse YOLO output and return ALL valid person crops without going out of bounds.
    """
    orig_w, orig_h = original_image.size
    in_w, in_h = resized_size
    preds = output_data[0]

    detections = []
    for det in preds:
        x_c, y_c, w_box, h_box = det[0:4].astype(float)
        obj_conf = float(det[4])
        class_probs = det[5:]
        class_id = int(np.argmax(class_probs))
        class_score = float(class_probs[class_id])
        conf = obj_conf * class_score

        if class_id == 0 and conf > conf_thresh:
            # scale to original image
            x1 = (x_c - w_box/2) * orig_w / in_w
            y1 = (y_c - h_box/2) * orig_h / in_h
            x2 = (x_c + w_box/2) * orig_w / in_w
            y2 = (y_c + h_box/2) * orig_h / in_h

            # clip to [0, orig_w] & [0, orig_h]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            # ensure valid box
            if x2 > x1 + 1 and y2 > y1 + 1:
                detections.append((conf, int(x1), int(y1), int(x2), int(y2)))

    if not detections:
        return []
    
    detections.sort(key=lambda x: x[0], reverse=True)
    keep = []
    for conf, x1, y1, x2, y2 in detections:
        area1 = (x2-x1)*(y2-y1)
        skip = False
        for _, kx1, ky1, kx2, ky2 in keep:
            # intersection
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
            inter = iw * ih
            area2 = (kx2-kx1)*(ky2-ky1)
            if inter / (area1 + area2 - inter + 1e-6) > iou_thresh:
                skip = True
                break
        if not skip:
            keep.append((conf, x1, y1, x2, y2))

    # Crop and save
    os.makedirs("crops", exist_ok=True)
    paths = []
    for idx, (_conf, x1, y1, x2, y2) in enumerate(keep, start=1):
        # Final sanity check
        if x1 < 0 or y1 < 0 or x2 > orig_w or y2 > orig_h:
            continue
        crop = original_image.crop((x1, y1, x2, y2))
        out = os.path.join("crops", f"crop_{idx}.png")
        crop.save(out)
        paths.append(out)

    return paths

def run_inference(image_path):
    """
    Full pipeline: read, preprocess, infer, postprocess → return crop paths.
    """
    # 1) Preprocess
    input_data, orig_img, size = preprocess_image(image_path)

    # 2) Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 3) Postprocess & return
    return postprocess(output_data, orig_img, size, conf_thresh=0.5, iou_thresh=0.4)
