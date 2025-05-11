import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile

model = whisper.load_model("base")

def record_voice(filename="command.wav", duration=3, fs=44100):
    print("🎙 Recording voice command...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    scipy.io.wavfile.write(filename, fs, recording)
    return filename

def transcribe_audio(filename="command.wav"):
    result = model.transcribe(filename)
    return result["text"].lower()