# File: send_image.py
import os
import base64
import zlib
from lora_setup import get_lora_radio

# === SETTINGS ===
IMAGE_PATH = "input.png"
USE_COMPRESSION = False  # Set to True for compressed mode
CHUNK_SIZE = 200  # How many bytes per LoRa packet

# === Load and (optionally) compress image ===
with open(IMAGE_PATH, "rb") as f:
    data = f.read()
    print(f"Original size: {len(data)} bytes")
    if USE_COMPRESSION:
        data = zlib.compress(data, level=9)
        print(f"Compressed size: {len(data)} bytes")

# === Encode to base16 (hex) ===
hex_data = base64.b16encode(data).decode("ascii")

# === Send in chunks ===
radio = get_lora_radio()
chunks = [hex_data[i:i+CHUNK_SIZE] for i in range(0, len(hex_data), CHUNK_SIZE)]

print(f"📡 Sending {len(chunks)} packets...")
for i, chunk in enumerate(chunks):
    print(f"[{i+1}/{len(chunks)}] Sending {len(chunk)} chars...")
    radio.send(bytes(chunk, "ascii"))

print("Done sending image.")
