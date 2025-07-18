import threading
from speech_to_text import transcribe_audio
from nl_to_text.ft_gpt2 import generate_flight_plan
import whisper
import warnings
import time
import sounddevice as sd
import numpy as np
import wave
from pynput import keyboard
import queue
import re
import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


model = None
model_loaded = threading.Event()
recording = False
audio_q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if recording:
        audio_q.put(indata.copy())

def record_audio_keypress(filename="audio.wav", samplerate=16000):
    global recording
    print("\n🔴 Press SPACE to start recording. Press SPACE again to stop. Press ESC to exit.")
    quit_listener = threading.Event()

    def on_press(key):
        global recording
        if key == keyboard.Key.space:
            if not recording:
                print("🎙️  Recording... Press SPACE again to stop.")
                recording = True
            else:
                print("⏹️  Stopping...")
                recording = False
                return False
        elif key == keyboard.Key.esc:
            print("Exiting.")
            quit_listener.set()
            return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback):
        while listener.is_alive() and not recording and not quit_listener.is_set():
            time.sleep(0.05)
        if quit_listener.is_set():
            listener.stop()
            return "quit"
        while listener.is_alive() and recording and not quit_listener.is_set():
            time.sleep(0.05)
        if quit_listener.is_set():
            listener.stop()
            return "quit"

    all_audio = []
    while not audio_q.empty():
        all_audio.append(audio_q.get())

    if all_audio:
        audio_np = np.concatenate(all_audio, axis=0)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes((audio_np * 32767).astype(np.int16).tobytes())
        print(f"✅ Saved audio to {filename}")
    else:
        print("No audio recorded.")

    if quit_listener.is_set():
        return "quit"

def load_model():
    global model
    print("🧠 Loading Whisper model in background...")
    warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
    model = whisper.load_model("small.en")
    model_loaded.set()
    print("✅ Whisper model loaded.")

def extract_latest_command_block(output):
    # Collect all command lines that contain () and end with [END]
    commands = re.findall(r'Command:\s*([a-zA-Z_]+\(\d+\));\s*\[END\]', output)

    if commands:
        # Remove duplicates and rebuild as one clean list
        cleaned_block = "[OUTPUT]\n" + "\n".join(commands)
        return cleaned_block
    else:
        # Fallback: just return whatever was generated
        return output.strip()
def map_to_natural_language_commands(cleaned_output):
    command_map = {
        "forward": "move the right joystick forward",
        "backward": "move the right joystick backward",
        "left": "move the right joystick to the left",
        "right": "move the right joystick to the right",
        "up": "move the left joystick upward",
        "down": "move the left joystick downward",
        "pan_left": "move the left joystick to the left",
        "pan_right": "move the left joystick to the right",
    }

    # Extract all command(time) patterns
    commands = re.findall(r'([a-z_]+)\((\d+)\)', cleaned_output)

    # Map to natural language
    natural_language = []
    for direction, seconds in commands:
        if direction in command_map:
            action = command_map[direction]
            natural_language.append(f"{action} for {seconds} seconds")
        else:
            natural_language.append(f"Unknown command: {direction}({seconds})")

    return natural_language


def main():
    audio_file = "audio.wav"
    threading.Thread(target=load_model).start()
    model_loaded.wait()
    while True:
        result = record_audio_keypress(audio_file)
        if result == "quit":
            print("Exiting...")
            break
        transcript = transcribe_audio(model, filename=audio_file)
        print(f"\n📝 Transcript: {transcript}\n")
        raw_output = generate_flight_plan(transcript)
        cleaned_output = extract_latest_command_block(raw_output)
        print(f"\n[USER INPUT] {transcript}")
        print(f"\n[CLEANED OUTPUT]\n{cleaned_output}")

        # Print natural language instructions
        lines = map_to_natural_language_commands(cleaned_output)
        for line in lines:
            print("-", line)

        # Speak the natural language instructions
        full_speech = ". Then, ".join(lines)
        speak(full_speech)



if __name__ == "__main__":
    main()
