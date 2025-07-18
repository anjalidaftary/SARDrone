from speech_to_text import transcribe_audio
from nl_to_text.ft_gpt2 import generate_flight_plan
import whisper
import threading
import warnings

model = None
model_loaded = threading.Event()

def load_model():
    global model
    print("🧠 Loading Whisper model in background...")
    warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
    model = whisper.load_model("small.en")
    model_loaded.set()
    print("✅ Whisper model loaded.")

def main():
    audio_file = "audio.wav"
    threading.Thread(target=load_model).start()
    model_loaded.wait()
    print("\nSpeak your drone command. Recording will start now...")
    transcript = transcribe_audio(model, filename=audio_file)
    print(f"\n📝 Transcript: {transcript}\n")
    flight_plan = generate_flight_plan(transcript)
    print(f"\n[USER INPUT] {transcript}")
    print(f"\n[OUTPUT]\n{flight_plan}")

if __name__ == "__main__":
    main()
