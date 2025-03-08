import assemblyai as aai
import os
from utils.config import ASSEMBLYAI_API_KEY


aai.settings.api_key = ASSEMBLYAI_API_KEY

transcriber = aai.Transcriber()

def transcribe_audio(audio_file):
    """
    Transcribes the given audio file using AssemblyAI API and returns the transcript text.

    """
    try:
        transcript = transcriber.transcribe(audio_file)
        if transcript.text:
            return transcript.text
        else:
            print(f"Transcription failed for {audio_file}.")
            return None
    except Exception as e:
        print(f"An error occurred while transcribing {audio_file}: {e}")
        return None