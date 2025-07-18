import os
import math
import zlib
import png
import base64

# Helper to clamp values between 0 and 255
def clip(value):
    return int(max(0, min(255, round(value))))

# Read image and convert to grayscale 2D list
def read_image_to_grayscale(image_path):
    reader = png.Reader(image_path)
    width, height, rows, info = reader.read()
    rows = list(rows)
    image = []

    if info.get('greyscale', False):
        for row in rows:
            image.append(list(row))
    else:
        channels = info.get('planes', 3)
        for row in rows:
            row = list(row)
            new_row = []
            for i in range(0, len(row), channels):
                R, G, B = row[i], row[i+1], row[i+2]
                gray = int(round(0.299 * R + 0.587 * G + 0.114 * B))
                new_row.append(gray)
            image.append(new_row)

    return image, width, height

# Nearest-neighbor resize
def resize_image(image, new_size):
    new_width, new_height = new_size
    orig_height = len(image)
    orig_width = len(image[0])
    new_image = []

    for j in range(new_height):
        orig_j = int(j * orig_height / new_height)
        row = []
        for i in range(new_width):
            orig_i = int(i * orig_width / new_width)
            row.append(image[orig_j][orig_i])
        new_image.append(row)

    return new_image

# Convert image to Base64 text: quantize, pack bits, compress, encode
def convert_image(image_path, bit_depth=4, size=(256, 256), dithering=False):
    assert 1 <= bit_depth <= 7, "bit_depth must be between 1 and 7"

    # 1) Load & resize
    image, orig_w, orig_h = read_image_to_grayscale(image_path)
    image = resize_image(image, size)
    width, height = size
    max_val = (1 << bit_depth) - 1

    # 2) Optional Floyd-Steinberg dithering
    if dithering:
        for y in range(height):
            for x in range(width):
                old_pixel = image[y][x]
                new_pixel_val = round(old_pixel * max_val / 255)
                new_pixel = int(new_pixel_val * (255 // max_val))
                error = old_pixel - new_pixel
                image[y][x] = new_pixel
 
                # Distribute error
                if x+1 < width:
                    image[y][x+1] = clip(image[y][x+1] + error * 7 / 16)
                if x-1 >= 0 and y+1 < height:
                    image[y+1][x-1] = clip(image[y+1][x-1] + error * 3 / 16)
                if y+1 < height:
                    image[y+1][x] = clip(image[y+1][x] + error * 5 / 16)
                if x+1 < width and y+1 < height:
                    image[y+1][x+1] = clip(image[y+1][x+1] + error * 1 / 16)

    # 3) Flatten and quantize pixels
    flat_pixels = []
    for row in image:
        for pixel in row:
            quant = pixel * max_val // 255
            flat_pixels.append(quant)

    # 4) Pack bits into bytes
    packed_bytes = bytearray()
    buffer = 0
    bits_filled = 0
    for val in flat_pixels:
        buffer = (buffer << bit_depth) | val
        bits_filled += bit_depth
        while bits_filled >= 8:
            bits_filled -= 8
            packed_bytes.append((buffer >> bits_filled) & 0xFF)
    if bits_filled > 0:
        buffer <<= (8 - bits_filled)
        packed_bytes.append(buffer & 0xFF)

    # 5) Compress and Base64-encode
    compressed = zlib.compress(packed_bytes)
    b64 = base64.b64encode(compressed).decode('ascii')
    print(f"Image converted successfully. Base64 length: {len(b64)}")
    return b64

def convert_binary(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    print(f"Binary image loaded: {len(data)} bytes")
    return data
