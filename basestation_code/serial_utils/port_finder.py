from serial.tools import list_ports
from logger import log_to_file

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

    # Fallback to interactive selection
    print("[INFO] Multiple serial ports found:")
    for i, port in enumerate(ports):
        print(f"  {i}: {port.device} - {port.description}")
        log_to_file(f"[INFO] {i}: {port.device} - {port.description}")
    idx = int(input("Enter the number of the port to use: "))
    return ports[idx].device