from src.data_collection.transcribe_audio import transcribe_audio
from src.preprocessing.clean_transcript import clean_transcript
import os
import logging


podcasts_folder = "data/raw/podcasts"
transcripts_folder = "data/raw/podcast_transcripts"
processed_folder = "data/processed"


os.makedirs(transcripts_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)

for audio_file in os.listdir(podcasts_folder):
    if audio_file.endswith((".mp3", ".wav", ".m4a")): 
        audio_path = os.path.join(podcasts_folder, audio_file)
        print(f"Transcribing {audio_file}...")
        
        transcript = transcribe_audio(audio_path)
        
        if transcript:
            file_name = os.path.splitext(audio_file)[0]
            
            raw_transcript_path = os.path.join(transcripts_folder, f"{file_name}_transcript.txt")
            
            with open(raw_transcript_path, "w") as file:
                file.write(transcript)
            print(f"Transcription successful! Transcript saved as {raw_transcript_path}.")
    

            cleaned_transcript = clean_transcript(transcript)

            cleaned_transcript_path = os.path.join(processed_folder, f"{file_name}_cleaned_transcript.txt")
            with open(cleaned_transcript_path, "w") as file:
                file.write(cleaned_transcript)
            print(f"Cleaned transcript saved as {cleaned_transcript_path}.")

        else:
            print(f"Transcription failed for {audio_file}.")


