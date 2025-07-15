# File: receive_image.py
import base64
import zlib
from lora_setup import get_lora_radio

radio = get_lora_radio()
received = ""

print("📡 Receiving packets...")
while True:
    packet = radio.receive(timeout=5.0)
    if packet is None:
        print("No more packets. Ending reception.")
        break
    try:
        received += packet.decode("ascii")
        print(f"Received {len(packet)} bytes.")
    except UnicodeDecodeError:
        print("Bad packet. Skipping.")

# === Decode ===
try:
    raw_data = base64.b16decode(received)
    try:
        decompressed = zlib.decompress(raw_data)
        with open("received_image.png", "wb") as f:
            f.write(decompressed)
        print("Saved decompressed image as received_image.png")
    except zlib.error:
        with open("received_image.png", "wb") as f:
            f.write(raw_data)
        print("Saved raw (uncompressed) image as received_image.png")
except Exception as e:
    print(f"Failed to decode image: {e}")
