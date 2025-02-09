# pip install flask twilio gTTS requests openai

from gtts import gTTS
import tempfile
import os

def synthesize_speech(text: str) -> str:
    """
    Converts the provided text to speech using Google Text-to-Speech.
    """
    tts = gTTS(text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    
    return temp_file.name  # Return path to the audio file
