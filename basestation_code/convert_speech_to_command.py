import threading
from speech_to_text import transcribe_audio
from sentence_transformer import parse_input
import whisper
import time
import sounddevice as sd
import numpy as np
import wave
from pynput import keyboard
import queue
import warnings

# Globals
model = None
model_loaded = threading.Event()
recording_done = threading.Event()
recording = False
audio_q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if recording:
        audio_q.put(indata.copy())

def record_audio_keypress(filename="audio.wav", samplerate=16000):
    global recording
    print("\n🔴 Press SPACE to start recording. Press SPACE again to stop.")

    def on_press(key):
        global recording
        if key == keyboard.Key.space:
            if not recording:
                print("🎙️  Recording... Press SPACE again to stop.")
                recording = True
            else:
                print("⏹️  Stopping...")
                recording = False
                return False  # Stop listener

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback):
        while listener.is_alive() and not recording:
            pass
        while listener.is_alive() and recording:
            pass

    # Collect audio data
    all_audio = []
    while not audio_q.empty():
        all_audio.append(audio_q.get())

    audio_np = np.concatenate(all_audio, axis=0)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((audio_np * 32767).astype(np.int16).tobytes())

    print(f"✅ Saved audio to {filename}")

def load_model():
    global model
    print("🧠 Loading Whisper model in background...")
    warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
    model = whisper.load_model("small.en")  # Use "small" for better accuracy
    model_loaded.set()
    print("✅ Whisper model loaded.")

def record_audio_async(filename="audio.wav"):
    record_audio_keypress(filename)
    recording_done.set()

def run_pipeline():
    audio_file = "audio.wav"

    threading.Thread(target=load_model).start()
    threading.Thread(target=record_audio_async, args=(audio_file,)).start()

    model_loaded.wait()
    recording_done.wait()

    transcript = transcribe_audio(model, filename=audio_file)
    print(f"\n📝 Transcript: {transcript}\n")

    actions = parse_input(transcript)
    print("📡 Parsed Commands:")
    if not actions:
        print("→ No valid commands found.")
    else:
        for action in actions:
            print(f"→ Command: {action['command']}, Value: {action['value']}, Confidence: {action['confidence']:.2f}")

if __name__ == "__main__":
    run_pipeline()
