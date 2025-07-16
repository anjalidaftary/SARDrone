'''
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

# === Send final token to signal end of stream ===
radio.send(b"END_OF_STREAM")

print("Done sending image.")
'''
import time
import math

def send_file(hex_data, handler):
    """
    Sends a base16 (hex) encoded string over LoRa using the provided handler.
    Each packet contains handler.max_packet_size bytes = max_packet_size hex characters.
    """
    packet_list = [hex_data[i:i + handler.max_packet_size] for i in range(0, len(hex_data), handler.max_packet_size)]
    
    print(f"Total packets to send: {len(packet_list)}")

    # set delay before sending ACK
    handler.rfm9x.ack_delay = 0.1
    # set node addresses
    handler.rfm9x.node = 1
    handler.rfm9x.destination = 2

    for packet in packet_list:
        print(packet.encode('ascii'))
        handler.rfm9x.send_with_ack(packet.encode('ascii'))
        time.sleep(0.1)

    return True