from lora_setup import get_lora_radio

def send_lora_payload(payload: bytes):
    """
    Sends a byte payload over LoRa.
    """
    try:
        rfm9x = get_lora_radio()
        print(f"Sending {len(payload)} bytes via LoRa...")
        rfm9x.send(payload)
        print("Payload sent.")
    except Exception as e:
        print(f"LoRa send error: {e}")
