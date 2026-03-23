import streamlit as st
import requests
import sounddevice as sd
import wavio
from openai import OpenAI
import os

# Correct API usage
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def record_audio(filename, duration, fs):
    st.write("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    wavio.write(filename, recording, fs, sampwidth=2)
    st.success("Recording complete")

st.title("🎤 Voice to Image Generator")

if st.button("Click here to speak"):
    try:
        audio_filename = "input.wav"
        record_audio(audio_filename, 5, 44100)

        # Speech to text
        with open(audio_filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        text = transcript.text
        st.write("📝 You said:", text)

        # Image generation
        response = client.images.generate(
            model="gpt-image-1",
            prompt=text,
            size="1024x1024"
        )

        image_url = response.data[0].url

        image_response = requests.get(image_url)
        image_path = "generated_image.jpg"

        with open(image_path, "wb") as f:
            f.write(image_response.content)

        st.image(image_path, caption="Generated Image")

    except Exception as e:
        st.error(f"Error: {e}")