from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite

# Load & allocate once at import time
_MODEL_PATH = "best-fp16.tflite"
_interpreter = tflite.Interpreter(model_path=_MODEL_PATH)
_interpreter.allocate_tensors()
_input_details  = _interpreter.get_input_details()
_output_details = _interpreter.get_output_details()

def preprocess_image(img_path, target_size=(640, 480)):
    img = Image.open(img_path).convert("RGB").resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0), img

def postprocess(output_data, original_image, conf_threshold=0.5):
    w, h = original_image.size
    crops = []
    # output_data shape assumed (1, N, 6) → [x_center, y_center, w, h, conf, class_id]
    for det in output_data[0]:
        x_c, y_c, bw, bh, conf, cls = det
        if conf >= conf_threshold and int(cls) == 0:  # person
            x1 = int((x_c - bw/2) * w)
            y1 = int((y_c - bh/2) * h)
            x2 = int((x_c + bw/2) * w)
            y2 = int((y_c + bh/2) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crops.append(original_image.crop((x1, y1, x2, y2)))
    return crops

def run_inference(img_path, output_dir="crops", conf_threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    input_data, orig = preprocess_image(img_path)
    _interpreter.set_tensor(_input_details[0]['index'], input_data)
    _interpreter.invoke()
    output_data = _interpreter.get_tensor(_output_details[0]['index'])
    crops = postprocess(output_data, orig, conf_threshold)
    paths = []
    for i, crop in enumerate(crops, start=1):
        fn = os.path.join(output_dir, f"crop_{i}.jpg")
        crop.save(fn)
        paths.append(fn)
    return paths
