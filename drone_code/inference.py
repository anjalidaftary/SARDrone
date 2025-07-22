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
    """Extract person crops from model output"""
    w, h = original_image.size
    crops = []

    output = output_data[0]  # should be shape (N, 6) or (N, 85) depending on model
    print(f"[DEBUG] Output shape: {output.shape}")

    for det in output:
        # If using YOLO-style model with [x, y, w, h, obj_conf, ...class_probs]
        if len(det) > 6:
            x_center, y_center, box_w, box_h = map(float, det[:4])
            obj_conf = float(det[4])
            class_probs = det[5:]

            class_id = int(np.argmax(class_probs))
            class_score = float(class_probs[class_id])
            conf = obj_conf * class_score

            print(f"[DEBUG] Det: conf={conf:.2f}, class_id={class_id}, obj_conf={obj_conf:.2f}, class_score={class_score:.2f}")
        else:
            # If already of form [x_center, y_center, w, h, conf, class_id]
            x_center, y_center, box_w, box_h, conf, class_id = det
            conf = float(conf)
            class_id = int(class_id)

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

def run_inference(image_path, output_dir="crops"):
    os.makedirs(output_dir, exist_ok=True)
    inp, orig, size = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    crops = postprocess(out, orig, size)
    paths = []
    for i, c in enumerate(crops, start=1):
        fn = os.path.join(output_dir, f"crop_{i}.jpg")
        c.save(fn); paths.append(fn)
    return paths
