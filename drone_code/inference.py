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

def postprocess(output_data, original_image, resized_size, conf_thresh=0.5, iou_thresh=0.4):
    """
    Parse YOLOv5-style output_data of shape (1,25200,85) and
    return ALL person crops (class_id==0, conf>conf_thresh).
    """
    orig_w, orig_h = original_image.size
    in_w,  in_h  = resized_size

    preds = output_data[0]  # (25200,85)
    detections = []

    # 1) collect raw boxes
    for det in preds:
        x_c, y_c, w_box, h_box = det[0:4].astype(float)
        obj_conf = float(det[4])
        class_probs = det[5:]
        class_id   = int(np.argmax(class_probs))
        class_score = float(class_probs[class_id])
        conf = obj_conf * class_score

        if class_id == 0 and conf > conf_thresh:
            # convert to pixels on original image
            x1 = int((x_c - w_box/2) * orig_w / in_w)
            y1 = int((y_c - h_box/2) * orig_h / in_h)
            x2 = int((x_c + w_box/2) * orig_w / in_w)
            y2 = int((y_c + h_box/2) * orig_h / in_h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            detections.append((conf, x1, y1, x2, y2))

    if not detections:
        return []

    # 2) optional NMS to remove highly overlapping boxes
    detections.sort(key=lambda x: x[0], reverse=True)
    keep = []
    for det in detections:
        _, x1, y1, x2, y2 = det
        area = (x2-x1)*(y2-y1)
        overlap = False
        for k in keep:
            _, kx1, ky1, kx2, ky2 = k
            # intersection
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
            inter = iw * ih
            union = area + (kx2-kx1)*(ky2-ky1) - inter
            if union > 0 and (inter/union) > iou_thresh:
                overlap = True
                break
        if not overlap:
            keep.append(det)

    # 3) crop and save each kept box
    os.makedirs("crops", exist_ok=True)
    paths = []
    for idx, det in enumerate(keep, start=1):
        _, x1, y1, x2, y2 = det
        crop = original_image.crop((x1, y1, x2, y2))
        out_path = os.path.join("crops", f"crop_{idx}.png")
        crop.save(out_path)
        paths.append(out_path)

    return paths

def run_inference(image_path):
    """
    Returns a list of paths to all cropped person images.
    """
    # 1) preprocess
    input_data, orig_image, size = preprocess_image(image_path)
    # 2) inference
    interpreter.set_tensor(_input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(_output_details[0]['index'])
    # 3) postprocess (now returns multiple)
    crops = postprocess(output_data, orig_image, size, conf_thresh=0.5, iou_thresh=0.4)
    return crops
