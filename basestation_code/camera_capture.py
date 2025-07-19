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