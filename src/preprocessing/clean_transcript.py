import re

def clean_transcript(transcript):
    # remove all non-alphanumeric characters
    transcript = re.sub(r'\W+', ' ', transcript)
    # convert to lowercase
    transcript = transcript.lower()

    return transcript