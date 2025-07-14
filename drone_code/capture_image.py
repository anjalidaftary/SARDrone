# File: capture_image.py
import subprocess
import os
from datetime import datetime

def capture_image(output_filename="input.png", width=640, height=480):
    # Create output path
    output_path = os.path.join(os.path.dirname(__file__), output_filename)

    try:
        print("Capturing image with libcamera...")
        subprocess.run([
            "libcamera-still",
            "-o", output_path,
            "--width", str(width),
            "--height", str(height),
            "--nopreview",
            "-t", "1000"  # capture after 1 second
        ], check=True)
        print(f"Image saved to {output_path}")
    except subprocess.CalledProcessError:
        print("Failed to capture image")

if __name__ == "__main__":
    capture_image()
