import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import os
import warnings

def record_audio(filename="audio.wav", duration=10, samplerate=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    write(filename, samplerate, audio)
    print(f"Saved recording to {filename}")

def transcribe_audio(model, filename="audio.wav"):
    warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
    print("Transcribing...")
    result = model.transcribe(filename)
    return result["text"]

if __name__ == "__main__":
    duration_seconds = 10  # Change this to record longer or shorter
    filename = "audio.wav"

    record_audio(filename=filename, duration=duration_seconds)
    transcribe_audio(filename=filename, model_size="small")  # or use "tiny"
