import time
import zlib
import base64
import binascii

# Text (Base64) pipeline
def send_file(b64_data, handler):
    """
    Sends a Base64-encoded string over LoRa in text mode.
    """
    handler.rfm9x.ack_delay   = 0.1
    handler.rfm9x.node        = 1
    handler.rfm9x.destination = 2

    packets = [
        b64_data[i : i + handler.max_packet_size]
        for i in range(0, len(b64_data), handler.max_packet_size)
    ]
    print(f"[SEND] {len(packets)} Base64 packets")
    for pkt in packets:
        handler.rfm9x.send_with_ack(pkt.encode('ascii'))
        time.sleep(0.1)
    return True

# Binary pipeline (hex‑encode before sending)
def send_binary(data_bytes, handler):
    """
    Compresses, hex‑encodes, and sends raw binary data over LoRa.
    """
    handler.rfm9x.ack_delay   = 0.1
    handler.rfm9x.node        = 1
    handler.rfm9x.destination = 2

    # First compress the raw bytes
    compressed = zlib.compress(data_bytes)
    # Then hex‑encode to get a printable string
    hex_str = binascii.hexlify(compressed).decode('ascii')

    packets = [
        hex_str[i : i + handler.max_packet_size]
        for i in range(0, len(hex_str), handler.max_packet_size)
    ]
    print(f"[SEND] {len(packets)} hex‑encoded packets ({len(hex_str)} chars)")
    for pkt in packets:
        handler.rfm9x.send_with_ack(pkt.encode('ascii'))
        time.sleep(0.1)
    return True
