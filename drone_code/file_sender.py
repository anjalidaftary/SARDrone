import time

def send_file(b64_data, handler):
    """
    Sends a Base64-encoded string over LoRa in text mode.
    """
    packets = [b64_data[i : i + handler.max_packet_size]
               for i in range(0, len(b64_data), handler.max_packet_size)]
    handler.rfm9x.ack_delay   = 0.1
    handler.rfm9x.node        = 1
    handler.rfm9x.destination = 2

    print(f"Total text packets: {len(packets)}")
    for pkt in packets:
        handler.rfm9x.send_with_ack(pkt.encode('ascii'))
        time.sleep(0.1)

    return True


def send_binary(data_bytes, handler):
    """
    Sends raw binary data (e.g., PNG or NPZ) over LoRa.
    """
    packets = [data_bytes[i : i + handler.max_packet_size]
               for i in range(0, len(data_bytes), handler.max_packet_size)]
    handler.rfm9x.ack_delay   = 0.1
    handler.rfm9x.node        = 1
    handler.rfm9x.destination = 2

    print(f"Total binary packets: {len(packets)}")
    for pkt in packets:
        handler.rfm9x.send_with_ack(pkt)
        time.sleep(0.1)

    return True