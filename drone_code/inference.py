# inference.py
import os
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# ---- IoU computation ----
def compute_iou(box1, box2):
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

# ---- Postprocessing: filter, NMS, crop & save ----
def postprocess(output_data, original_image, resized_size, conf_thresh=0.5, iou_thresh=0.4):
    # ensure output folder exists
    os.makedirs("crops", exist_ok=True)

    orig_w, orig_h = original_image.size
    in_w, in_h = resized_size
    preds = output_data[0]

    # 1) collect all detections
    detections = []
    for det in preds:
        x_c, y_c, w_box, h_box = det[0:4].astype(float)
        obj_conf = float(det[4])
        class_probs = det[5:]
        class_id = int(np.argmax(class_probs))
        class_score = float(class_probs[class_id])
        conf = obj_conf * class_score

        # only keep "person" class (id==0) above threshold
        if class_id == 0 and conf > conf_thresh:
            # scale back to original image size
            x1 = (x_c - w_box/2) * orig_w / in_w
            y1 = (y_c - h_box/2) * orig_h / in_h
            x2 = (x_c + w_box/2) * orig_w / in_w
            y2 = (y_c + h_box/2) * orig_h / in_h
            detections.append([int(x1), int(y1), int(x2), int(y2), conf])

    print(f"[DEBUG] raw detections: {len(detections)}")

    # 2) non-maximum suppression
    detections.sort(key=lambda x: x[4], reverse=True)
    keep = []
    for box in detections:
        if all(compute_iou(box, kept) < iou_thresh for kept in keep):
            keep.append(box)

    print(f"[DEBUG] kept {len(keep)} after NMS")

    # 3) crop & save
    paths = []
    for idx, (x1, y1, x2, y2, conf) in enumerate(keep):
        # clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)
        crop = original_image.crop((x1, y1, x2, y2))
        out_path = os.path.join("crops", f"crop_{idx}.jpg")
        crop.save(out_path)
        print(f"[DEBUG] saved {out_path}")
        paths.append(out_path)

    return paths

# ---- Interpreter setup & run ----
_interpreter = None
_input_details = None
_output_details = None

def run_inference(image_path,
                  model_path="yolov5n.tflite",
                  input_size=(320, 320),
                  conf_thresh=0.5,
                  iou_thresh=0.4):
    """
    1) Load TFLite model once
    2) Preprocess input image
    3) Run inference
    4) Postprocess outputs → return list of crop file paths
    """
    global _interpreter, _input_details, _output_details

    # 1) load model if needed
    if _interpreter is None:
        model_file = os.path.join(os.path.dirname(__file__), model_path)
        _interpreter = tflite.Interpreter(model_path=model_file)
        _interpreter.allocate_tensors()
        _input_details  = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()

    # 2) load & preprocess image
    orig = Image.open(image_path).convert("RGB")
    in_w, in_h = input_size
    resized = orig.resize((in_w, in_h), Image.BILINEAR)
    input_data = np.asarray(resized, dtype=np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    # 3) inference
    _interpreter.set_tensor(_input_details[0]['index'], input_data)
    _interpreter.invoke()
    output_data = _interpreter.get_tensor(_output_details[0]['index'])  # shape [1,N,5+classes]

    # 4) postprocess & return crops
    return postprocess(
        output_data,
        original_image=orig,
        resized_size=(in_w, in_h),
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
    )