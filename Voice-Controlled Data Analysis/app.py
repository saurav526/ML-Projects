import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from openai import OpenAI
import os

# Initialize client using Streamlit secrets (for cloud) or .env (for local)
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# Load dataset safely
if not os.path.exists("sales.csv"):
    st.error("❌ sales.csv not found. Please upload your data file.")
    st.stop()

df = pd.read_csv("sales.csv")
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # convert once at load time

# Speech to text
def speech_to_text(audio_bytes):
    with open("input.wav", "wb") as f:
        f.write(audio_bytes.read())
    with open("input.wav", "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript.text.lower()

# Rule-based intent detection
def detect_intent(text):
    if "trend" in text:
        return "trend"
    elif "total" in text:
        return "total"
    elif "region" in text:
        return "region"
    elif "product" in text:
        return "product"
    else:
        return "unknown"

# Analysis functions
def show_trend():
    trend = df.groupby('date')['sales'].sum()
    fig, ax = plt.subplots()
    trend.plot(ax=ax)
    ax.set_title("Sales Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    st.pyplot(fig)
    plt.close(fig)

def show_total():
    total = df['sales'].sum()
    st.write("💰 Total Sales:", total)

def show_region_sales():
    region_sales = df.groupby('region')['sales'].sum()
    st.write("📊 Sales by Region:")
    st.write(region_sales)

def show_product_sales():
    product_sales = df.groupby('product')['sales'].sum()
    st.write("📦 Sales by Product:")
    st.write(product_sales)

# Streamlit UI
st.title("🎤 Voice-Based Data Analysis")

st.write("Try saying:")
st.write("- Show sales trend")
st.write("- Total sales")
st.write("- Sales by region")
st.write("- Sales by product")

# ✅ Use Streamlit's built-in mic widget (works on cloud, no sounddevice needed)
audio_bytes = st.audio_input("🎤 Click to record your query")

if audio_bytes:
    with st.spinner("Transcribing..."):
        text = speech_to_text(audio_bytes)

    st.write("📝 You said:", text)

    intent = detect_intent(text)

    if intent == "trend":
        show_trend()
    elif intent == "total":
        show_total()
    elif intent == "region":
        show_region_sales()
    elif intent == "product":
        show_product_sales()
    else:
        st.error("❌ Could not understand command. Try saying 'trend', 'total', 'region', or 'product'.")