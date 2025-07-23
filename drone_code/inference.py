import os
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

# Load TFLite model
interpreter = Interpreter(model_path=os.path.join(os.path.dirname(__file__), "yolov5n.tflite"))
interpreter.allocate_tensors()

# Get model I/O details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']  # [1, height, width, channels]


def preprocess_image(img_path):
    """
    Open an image, resize to model's input, normalize, and return the tensor + original PIL image.
    """
    img = Image.open(img_path).convert("RGB")
    _, inp_h, inp_w, _ = input_shape
    resized = img.resize((inp_w, inp_h), Image.BILINEAR)
    img_np = np.array(resized, dtype=np.float32) / 255.0
    img_np = np.expand_dims(img_np, axis=0)
    return img_np.astype(np.float32), img


def postprocess(output_data, original_image, conf_thresh=0.5):
    """
    Parse model output of shape [1, N, 6], extract person detections and return list of crop file paths.
    """
    w, h = original_image.size
    crops = []
    paths = []

    for det in output_data[0]:  # [x_center, y_center, box_w, box_h, conf, class_id]
        x_c, y_c, bw, bh, conf, cls = det
        if conf > conf_thresh and int(cls) == 0:
            # Convert to pixel coordinates
            x1 = int((x_c - bw/2) * w)
            y1 = int((y_c - bh/2) * h)
            x2 = int((x_c + bw/2) * w)
            y2 = int((y_c + bh/2) * h)
            # Clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            # Crop
            crop = original_image.crop((x1, y1, x2, y2))
            crops.append(crop)

    # Ensure output directory
    os.makedirs("crops", exist_ok=True)
    for idx, crop in enumerate(crops, start=1):
        out_path = os.path.join("crops", f"crop_{idx}.jpg")
        crop.save(out_path)
        paths.append(out_path)
    return paths


def run_inference(image_path, conf_thresh=0.5):
    """
    1) Preprocess image
    2) Run TFLite model
    3) Postprocess and save crops
    Returns list of crop file paths.
    """
    # 1) preprocess
    input_data, orig = preprocess_image(image_path)
    # 2) inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # Expect [1, N, 6]
    # 3) postprocess + save
    return postprocess(output_data, orig, conf_thresh=conf_thresh)