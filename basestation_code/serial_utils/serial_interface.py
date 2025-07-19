import time
import serial
import threading
import time
import re

from script_handler    import ScriptRunner
from reconstructor     import reconstruct_text, reconstruct_binary
from logger            import log_to_file
from .port_finder      import find_adafruit_port

from pathlib import Path
BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR.parent / "terminal.txt"

class SerialInterface:
    FILE_TRANSFER_GAP = 1.0  # seconds

    def __init__(self, port=None, baudrate=115200, timeout=1):
        self.port     = port if port is not None else find_adafruit_port()
        self.baudrate = baudrate
        self.timeout  = timeout
        self.ser      = None
        self.stop_event    = threading.Event()
        self.reader_thread = None

    def connect(self):
        # Ensure any prior handle is closed
        if self.ser and self.ser.is_open:
            self.ser.close()
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"[INFO] Connected to {self.port} at {self.baudrate} baud.")
            log_to_file(f"[INFO] Connected to {self.port} at {self.baudrate} baud.")
        except serial.SerialException as e:
            print(f"[ERROR] Could not open serial port {self.port}: {e}")
            log_to_file(f"[ERROR] Could not open serial port {self.port}: {e}")
            raise

    def start_reader(self):
        def read_from_port():
            buf = b""
            while not self.stop_event.is_set():
                try:
                    if self.ser.in_waiting:
                        buf += self.ser.read(self.ser.in_waiting)
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        try:
                            text = line.decode('utf-8')
                            print(f"[FEATHER] {text}")
                            log_to_file(f"[FEATHER] {text}")
                        except UnicodeDecodeError:
                            # treat as raw/binary data; just stash it
                            log_to_file("[FEATHER] [BINARY DATA]")
                            buf += line
                    time.sleep(0.05)
                except Exception as e:
                    print(f"[ERROR] Serial read error: {e}")
                    log_to_file(f"[ERROR] Serial read error: {e}")
                    break

        self.connect()
        self.reader_thread = threading.Thread(target=read_from_port, daemon=True)
        self.reader_thread.start()

    def send_command(self, cmd):
        if not (self.ser and self.ser.is_open):
            print("[ERROR] Serial port not open.")
            log_to_file("[ERROR] Serial port not open.")
            return
        try:
            print(f"[SEND] {cmd}")
            log_to_file(f"[SEND] {cmd}")
            self.ser.write((cmd + "\r\n").encode('utf-8'))
            self.ser.flush()
        except serial.SerialException as e:
            print(f"[ERROR] Failed to send command: {e}")
            log_to_file(f"[ERROR] Failed to send command: {e}")
            self.close()

    def reconstruct_image(self, bit_depth=4, size=(64,64)):
        try:
            # Choose your routine
            # Base64 text
            reconstruct_text(
                terminal_file=str(LOG_FILE),
                output_path="reconstructed_text.png",
                bit_depth=bit_depth,
                size=size
            )
        except Exception:
            # If text failed, try hex
            try:
                reconstruct_binary(
                    terminal_file=str(LOG_FILE),
                    output_path="reconstructed_binary.png",
                    bit_depth=bit_depth,
                    image_size=size
                )
            except Exception as e:
                print(f"[ERROR] Reconstruction failed both pipelines: {e}")
                log_to_file(f"[ERROR] Reconstruction failed: {e}")

    def interactive_mode(self):
        print(">> Type commands to send to the Feather (type 'exit' to quit):")
        try:
            while True:
                cmd = input(">> ").strip()
                if cmd.lower() in {'exit', 'quit'}:
                    print("[INFO] Exiting interactive mode...")
                    log_to_file("[INFO] Exiting interactive mode...")
                    break

                if cmd.upper().startswith("SCRIPT"):
                    parts = cmd.split()
                    if len(parts) >= 2:
                        filename = parts[1]
                        script_runner = ScriptRunner(self.send_command)
                        script_runner.run_script(filename)
                    else:
                        print("[ERROR] SCRIPT command requires a filename.")
                        log_to_file("[ERROR] SCRIPT command requires a filename.")

                elif cmd.upper().startswith("DISPLAY"):
                    self.extract_and_display_image()

                else:
                    self.send_command(cmd)
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt received. Exiting...")
            log_to_file("[INFO] Keyboard interrupt received. Exiting...")
        finally:
            self.close()

    def camera_capture(self):
        try:
            self.send_command("CAMERA text")
            with open(LOG_FILE, "r") as f:
                while "SCREENSHOT SENT" not in f.read():
                    time.sleep(0.5)
        finally:
            self.close()

    def close(self):
        self.stop_event.set()
        if self.reader_thread:
            self.reader_thread.join()
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[INFO] Serial port closed.")
            log_to_file("[INFO] Serial port closed.")
