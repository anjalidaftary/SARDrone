from serial_utils.serial_interface import SerialInterface
from pathlib import Path
from reconstructor import extract_chunks, reconstruct_binary, reconstruct_text

LOG_PATH = Path(__file__).resolve().parent / "terminal.txt"

def camera_capture():
    # connect feather
    print("Basestation online. Starting serial interface for camera capture...")
    serial_interface = SerialInterface()
    serial_interface.connect()
    serial_interface.start_reader()
    serial_interface.camera_capture()

    # construct image
    b64, hx = extract_chunks()
    if b64:
        reconstruct_text()
    elif hx:
        reconstruct_binary()
    else:
        print("[✗] No image data found in terminal.txt")

    # clean up
    with open(LOG_PATH, "w"):
        pass

def detect_capture():
    """
    Sends the DETECT command to the drone, then automatically
    reconstructs any cropped-person images that come back.
    """
    from serial_utils.serial_interface import SerialInterface
    from reconstructor import extract_chunks, reconstruct_text, reconstruct_binary
    from pathlib import Path
    import time

    DETECT_LOG = Path(__file__).resolve().parent / "terminal.txt"
    OUT_DIR   = Path(__file__).resolve().parent / "detect_crops"

    # 1) spin up serial I/O
    print("Starting DETECT sequence over serial…")
    serial_interface = SerialInterface()
    serial_interface.connect()
    serial_interface.start_reader()

    try:
        # 2) send the command
        serial_interface.send_command("DETECT")

        # 3) wait for the Pi to finish streaming
        timeout = time.time() + 60  # seconds
        while True:
            data = DETECT_LOG.read_text()
            if "[RESULT] DETECTION COMPLETE" in data:
                break
            if time.time() > timeout:
                print("[✗] Timeout waiting for DETECT COMPLETE")
                return
            time.sleep(0.5)

    finally:
        # 4) tear down serial I/O
        serial_interface.close()

    # 5) reconstruct any base64 or hex chunks
    b64, hx = extract_chunks()
    OUT_DIR.mkdir(exist_ok=True)

    if b64:
        # send an output directory so it doesn’t stomp your screenshot file
        reconstruct_text(output_dir=str(OUT_DIR))
    elif hx:
        reconstruct_binary(output_dir=str(OUT_DIR))
    else:
        print("[✗] No DETECT image data found in terminal.txt")

    # 6) clear the log for next time
    DETECT_LOG.write_text("")
    print(f"[✓] All DETECT crops written to {OUT_DIR}/")
#camera_capture()
detect_capture()