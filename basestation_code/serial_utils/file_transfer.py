import os
import re
import zlib
import base64
import png
from logger import log_to_file

def reconstruct_from_text(log_path, output_path="reconstructed_text.png",
                          bit_depth=4, image_size=(128,128)):
    """
    Pull Base64 fragments out of the log, decode, unpack pixels, and write a PNG.
    """
    # 1) Read & extract only valid Base64 lines
    b64_chunks = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"\[RECEIVED #[0-9]+\] \[\d+ bytes\]: (.+)", line)
            if not m: 
                continue
            payload = m.group(1).strip()
            if re.fullmatch(r"[A-Za-z0-9+/=]+", payload):
                b64_chunks.append(payload)
    if not b64_chunks:
        raise ValueError("No Base64 payload found in log.")

    b64_str = "".join(b64_chunks)

    # 2) Decode & decompress
    compressed = base64.b64decode(b64_str)
    raw        = zlib.decompress(compressed)

    # 3) Unpack bit‑packed pixels
    width, height = image_size
    total_pixels  = width * height
    max_val       = (1 << bit_depth) - 1
    scale         = 255 // max_val

    pixels = []
    buffer = 0
    bits   = 0
    for byte in raw:
        buffer = (buffer << 8) | byte
        bits  += 8
        while bits >= bit_depth and len(pixels) < total_pixels:
            bits   -= bit_depth
            val     = (buffer >> bits) & max_val
            pixels.append(val * scale)
        buffer &= (1 << bits) - 1

    if len(pixels) < total_pixels:
        raise ValueError(f"Incomplete image: got {len(pixels)} pixels, expected {total_pixels}")

    # 4) Reshape and write
    image = [pixels[i*width:(i+1)*width] for i in range(height)]
    with open(output_path, "wb") as f:
        writer = png.Writer(width, height, greyscale=True, bitdepth=8)
        writer.write(f, image)

    log_to_file(f"[FEATHER] TEXT reconstruction complete. Saved to '{output_path}'")
    print(f"[FEATHER] TEXT reconstruction complete. Saved to '{output_path}'")


def reconstruct_from_hex(log_path, output_path="reconstructed_hex.png",
                         bit_depth=4, image_size=(128,128)):
    """
    Pull hex fragments out of the log, decode, unpack pixels, and write a PNG.
    (This replaces your old reconstruct_image_from_hex.)
    """
    hex_chunks = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"\[RECEIVED #[0-9]+\] \[\d+ bytes\]: ([0-9A-Fa-f]+)", line)
            if m:
                hex_chunks.append(m.group(1).strip())
    if not hex_chunks:
        raise ValueError("No hex payload found in log.")

    hex_str     = "".join(hex_chunks)
    compressed  = bytes.fromhex(hex_str)
    raw         = zlib.decompress(compressed)

    # Unpack just like text‑mode
    width, height = image_size
    total_pixels  = width * height
    max_val       = (1 << bit_depth) - 1
    scale         = 255 // max_val

    pixels = []
    buffer = 0
    bits   = 0
    for byte in raw:
        buffer = (buffer << 8) | byte
        bits  += 8
        while bits >= bit_depth and len(pixels) < total_pixels:
            bits   -= bit_depth
            val     = (buffer >> bits) & max_val
            pixels.append(val * scale)
        buffer &= (1 << bits) - 1

    if len(pixels) < total_pixels:
        raise ValueError(f"Incomplete image: got {len(pixels)} pixels, expected {total_pixels}")

    image = [pixels[i*width:(i+1)*width] for i in range(height)]
    with open(output_path, "wb") as f:
        writer = png.Writer(width, height, greyscale=True, bitdepth=8)
        writer.write(f, image)

    log_to_file(f"[FEATHER] HEX reconstruction complete. Saved to '{output_path}'")
    print(f"[FEATHER] HEX reconstruction complete. Saved to '{output_path}'")


def reconstruct_image(log_path="terminal.txt", **kwargs):
    """
    Auto‑detects Base64 vs hex in the log and calls the appropriate function.
    """
    # Peek at file to see which payload we have
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    if re.search(r"[A-Za-z0-9+/=]{10,}", text):
        reconstruct_from_text(log_path=log_path, **kwargs)
    elif re.search(r"[0-9A-Fa-f]{10,}", text):
        reconstruct_from_hex(log_path=log_path, **kwargs)
    else:
        raise ValueError("No recognizable image payload in log.")
