import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
import tempfile

# Load environment variables
load_dotenv()

# Initialize the Groq client
try:
    client = Groq(api_key=os.getenv("TOKEN"))
except Exception as e:
    st.error(f"Error initializing Groq client: {e}")
    st.stop()

def transcribe_audio(audio_file):
    """Transcribes the given audio file and returns the transcription data."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(audio_file.read())
            audio_path = temp_audio_file.name

        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3-turbo",
                prompt="Specify context or spelling",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                language="en",
                temperature=0.0
            )
        data = transcription.to_dict()["words"]
        return data
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

def main():
    st.title("Speech to Text Transcription")

    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/mp3")
        st.subheader("Transcription:")

        transcription_data = transcribe_audio(uploaded_file)

        if transcription_data:
            markdown_output = ""
            for word_data in transcription_data:
                word = word_data.get('word', '')
                start = word_data.get('start', '')
                end = word_data.get('end', '')
                markdown_output += f"* **{word}**: {start:.2f} - {end:.2f} seconds\n"
            st.markdown(markdown_output)

if __name__ == "__main__":
    main()
