from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import os

#https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter

interpreter = tflite.Interpreter(model_path="best-fp16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((640, 480))
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = np.expand_dims(img_np, axis=0) 
    return img_np, img

def postprocess(output_data, original_image):
    h, w = original_image.size
    crops = []
    output = output_data[0]

    for det in output:
        x_center, y_center, box_w, box_h, conf, class_id = det

        if conf > 0.50 and int(class_id) == 0:  # class 0 = person

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

def run_inference(image_path):
    input_data, original_image = preprocess_image(image_path)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    crops = postprocess(output_data, original_image)
    for i, crop in enumerate(crops):
        crop.save(f"crop_{i+1}.jpg")

run_inference("IMAGE.jpg")