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

def postprocess(preds, orig_img, resized_size, conf_thresh=0.4, iou_thresh=0.5):
    boxes = []
    img_w, img_h = orig_img.size
    in_w, in_h = resized_size
    for x, y, w, h, conf, cls in preds[0]:
        if conf < conf_thresh or int(cls) != 0:
            continue
        # scale back to original
        sx, sy = img_w/in_w, img_h/in_h
        x1 = int((x - w/2)*sx); y1 = int((y - h/2)*sy)
        x2 = int((x + w/2)*sx); y2 = int((y + h/2)*sy)
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(img_w,x2), min(img_h,y2)
        boxes.append([x1,y1,x2,y2,conf])
    # nms
    boxes.sort(key=lambda b: b[4], reverse=True)
    final = []
    while boxes:
        best = boxes.pop(0); final.append(best)
        boxes = [b for b in boxes if compute_iou(best,b) < iou_thresh]
    # crop
    return [ orig_img.crop((x1,y1,x2,y2)) for x1,y1,x2,y2,_ in final ]

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
