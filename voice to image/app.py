import streamlit as st
import sounddevice as sd
import wavio
import base64
import tempfile
import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def record_audio(filename, duration, fs):
    try:
        sd.check_input_settings(samplerate=fs, channels=1)
    except Exception as e:
        st.error(f"Microphone not available: {e}")
        st.stop()

    st.write("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)  # ✅ Mono
    sd.wait()
    wavio.write(filename, recording, fs, sampwidth=2)
    st.success("Recording complete")

st.title("🎤 Voice to Image Generator")

if st.button("Click here to speak"):
    try:
        # ✅ Unique temp files
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            audio_filename = tmp_audio.name
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
            image_path = tmp_img.name

        record_audio(audio_filename, 5, 44100)

        # Speech to text
        with open(audio_filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        text = transcript.text
        
        # ✅ Guard against empty transcription
        if not text.strip():
            st.warning("No speech detected. Please try again.")
            st.stop()

        st.write("📝 You said:", text)

        # Image generation - ✅ use b64_json, not url
        response = client.images.generate(
            model="gpt-image-1",
            prompt=text,
            size="1024x1024",
            response_format="b64_json"
        )

        image_data = base64.b64decode(response.data[0].b64_json)
        with open(image_path, "wb") as f:
            f.write(image_data)

        st.image(image_path, caption="Generated Image")

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        # ✅ Clean up temp files
        for path in [audio_filename, image_path]:
            if os.path.exists(path):
                os.remove(path)