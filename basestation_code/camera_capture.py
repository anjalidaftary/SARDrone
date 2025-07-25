from serial_utils.serial_interface import SerialInterface
from pathlib import Path
from reconstructor import extract_chunks, reconstruct_binary, reconstruct_text
import time
import shutil

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
    reconstructs any cropped-person images that come back,
    and moves each reconstructed PNG into detect_crops/crop_1.png, etc.
    """
    DETECT_LOG = LOG_PATH
    OUT_DIR   = DETECT_LOG.parent / "detect_crops"

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
            data = DETECT_LOG.read_text(encoding="utf-8", errors="replace")
            if "[RESULT] DETECTION COMPLETE" in data:
                break
            if time.time() > timeout:
                print("[✗] Timeout waiting for DETECT COMPLETE")
                return
            time.sleep(0.5)

    finally:
        # 4) tear down serial I/O
        serial_interface.close()

    # 5) clear any old reconstructions
    for old in DETECT_LOG.parent.glob("reconstructed_*.png"):
        old.unlink()

    # 6) reconstruct any base64 or hex chunks
    b64, hx = extract_chunks()
    if b64:
        reconstruct_text()       # no args
    elif hx:
        reconstruct_binary()     
    else:
        print("[✗] No DETECT image data found in terminal.txt")
        return

    # 7) move each reconstructed image into detect_crops/
    OUT_DIR.mkdir(exist_ok=True)
    recon_files = sorted(DETECT_LOG.parent.glob("reconstructed_*.png"))
    if not recon_files:
        print("[✗] Reconstruction succeeded but no files found.")
        return

    for idx, src in enumerate(recon_files, start=1):
        dst = OUT_DIR / f"crop_{idx}.png"
        # if a file with that name already exists, overwrite it
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
        print(f"[✓] Moved {src.name} → detect_crops/{dst.name}")

    # 8) clear the log for next time
    DETECT_LOG.write_text("")
    print(f"[✓] All DETECT crops written to {OUT_DIR}/")
#camera_capture()
detect_capture()