from lora_setup import get_lora_radio

def receive_lora_payload(timeout=10.0):
    """
    Receives a LoRa payload within a timeout (seconds).
    Returns: bytes or None
    """
    try:
        rfm9x = get_lora_radio()
        print("Waiting for incoming payload...")
        packet = rfm9x.receive(timeout=timeout)
        if packet is None:
            print("No payload received.")
            return None
        print(f"Received {len(packet)} bytes.")
        return packet
    except Exception as e:
        print(f"LoRa receive error: {e}")
        return None
