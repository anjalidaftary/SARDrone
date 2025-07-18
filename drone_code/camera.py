import os
import subprocess
import time

def capture_photo(
    save_directory="img",
    filename="input.png",
    width=64,
    height=64,  
    fmt="png"    
):
    """
    Capture a photo using libcamera-still with specified resolution and encoding.
    """
    # Create folder if needed
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    photo_path = os.path.join(save_directory, filename)

    cmd = [
        "libcamera-still",
        "-o", photo_path,
        "--width",  str(width),
        "--height", str(height),
        "--encoding", fmt,
        "-t", "100"
    ]

    try:
        print(f"Capturing image to {photo_path} at {width}x{height} as {fmt}...")
        subprocess.run(cmd, check=True)
        time.sleep(0.2)
        print(f"Image saved to {photo_path}")
        return photo_path
    except subprocess.CalledProcessError as e:
        print(f"[capture_photo] libcamera-still failed: {e}")
        return None