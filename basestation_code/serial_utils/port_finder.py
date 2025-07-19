from serial.tools import list_ports
from logger import log_to_file
from pathlib import Path
import re

def find_adafruit_port():
    ports = list(list_ports.comports())
    if not ports:
        raise IOError("No serial ports found.")

    # Filter for anything likely our board
    candidates = [p for p in ports if
                  any(token in p.description for token in ("Adafruit","Feather")) or
                  "usb" in p.device.lower()]

    if len(candidates) == 1:
        return candidates[0].device
    elif len(candidates) > 1:
        ports = candidates  # prompt only on the filtered list

    # Specific search for Feather
    LOG_PATH = Path(__file__).resolve().parent.parent / "terminal.txt"
    pattern = r"\b(\d+):\s*/dev/cu\.usbmodem\d+.*Feather RP2040 RFM"
    with open(LOG_PATH, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                print(f"[DEBUG] Match found.")
                return ports[int(match.group(1))].device

    # Fallback to interactive selection
    print("[INFO] Multiple serial ports found:")
    for i, port in enumerate(ports):
        print(f"  {i}: {port.device} - {port.description}")
        log_to_file(f"[INFO] {i}: {port.device} - {port.description}")
    idx = int(input("Enter the number of the port to use: "))
    return ports[idx].device