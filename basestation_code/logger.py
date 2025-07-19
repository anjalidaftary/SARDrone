from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "terminal.txt"

def log_to_file(message):
    try:
        print(f"[DEBUG] Logging message to {LOG_PATH}")  # ← Add this
        with open(str(LOG_PATH), "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}    {message}\n")
    except Exception as e:
        print(f"[ERROR] Logging failed: {e}")
