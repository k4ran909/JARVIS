
from random import randint
from gtts import gTTS
from io import BytesIO
import tempfile
import os
import base64
import io

def text_to_speech_local(text, lang="en"):
    """Converts text to speech using gTTS and streams audio to the browser."""
    
    # Generate TTS audio using gTTS
    tts = gTTS(text=text, lang=lang)
    
    # Create a temporary file to store audio as MP3
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        audio_path = temp_audio.name

    # Save audio to the temporary file
    tts.save(audio_path)

    # Read the audio content as bytes into a BytesIO stream
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    
    # Delete the temporary file after reading
    os.remove(audio_path)


    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

    return audio_base64 

if __name__ == "__main__":
    audio_io = text_to_speech_local("hello how are you.")
    print(audio_io)