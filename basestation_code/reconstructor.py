import os
import re
import zlib
import base64
import binascii
import png

LOG = "../terminal.txt"
OUT_B64 = "reconstructed_text.png"
OUT_BIN = "reconstructed_binary.png"

def extract_chunks():
    """
    Pulls all payloads from lines like:
      [RECEIVED #2] [32 bytes]: eJxjY...
      [RECEIVED #5] [120 bytes]: 4d5a90...
    Returns two strings: base64_data, hex_data
    """
    b64_chunks = []
    hex_chunks = []
    pattern = re.compile(r"\[RECEIVED #[0-9]+\] \[\d+ bytes\]: (.+)$")
    with open(LOG, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if not m: continue
            data = m.group(1).strip()
            # Base64 is A–Z,a–z,0–9,+,/ and ends in '=' padding
            if re.fullmatch(r"[A-Za-z0-9+/=]+", data):
                b64_chunks.append(data)
            # Hex is 0–9,a–f (even length)
            elif re.fullmatch(r"[0-9A-Fa-f]+", data):
                hex_chunks.append(data)
    return "".join(b64_chunks), "".join(hex_chunks)

def reconstruct_text(bit_depth=4, size=(64,64)):
    b64_str, _ = extract_chunks()
    if not b64_str:
        print("[✗] No Base64 data found.")
        return

    print("[DEBUG] Base64 string length:", len(b64_str))
    compressed = base64.b64decode(b64_str)
    print("[DEBUG] Compressed size (base64-decoded):", len(compressed))

    raw = zlib.decompress(compressed)
    print("[DEBUG] Decompressed raw byte length:", len(raw))

    width, height = size
    max_val = (1 << bit_depth) - 1
    scale = 255 // max_val
    pixels = []
    buffer = 0
    bits = 0
    for byte in raw:
        buffer = (buffer << 8) | byte
        bits  += 8
        while bits >= bit_depth and len(pixels) < width*height:
            bits -= bit_depth
            val  = (buffer >> bits) & max_val
            pixels.append(val * scale)
        buffer &= (1 << bits) - 1

    print("[DEBUG] Total pixels reconstructed:", len(pixels))
    print("[DEBUG] Sample pixel values:", pixels[:10])  # check if they are all 0

    if len(pixels) < width*height:
        print(f"[✗] Incomplete: got {len(pixels)} pixels")
        return

    img = [ pixels[i*width:(i+1)*width] for i in range(height) ]
    with open(OUT_B64, 'wb') as f:
        writer = png.Writer(width, height, greyscale=True, bitdepth=8)
        writer.write(f, img)
    print(f"[✓] Text‑mode image saved to {OUT_B64}")

def reconstruct_binary():
    _, hex_str = extract_chunks()
    if not hex_str:
        print("[✗] No hex data found.")
        return

    print("[DEBUG] Hex string length:", len(hex_str))
    compressed = binascii.unhexlify(hex_str)
    print("[DEBUG] Compressed binary size:", len(compressed))

    raw = zlib.decompress(compressed)
    print("[DEBUG] Decompressed PNG size:", len(raw))
    print("[DEBUG] First 16 bytes of raw PNG:", raw[:16])

    with open(OUT_BIN, 'wb') as f:
        f.write(raw)
    print(f"[✓] Binary‑mode image saved to {OUT_BIN}")

if __name__ == "__main__":
    # prioritize Base64 if present
    b64, hx = extract_chunks()
    if b64:
        reconstruct_text()
    elif hx:
        reconstruct_binary()
    else:
        print("[✗] No image data found in terminal.txt")