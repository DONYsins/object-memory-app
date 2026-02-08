import os
import requests
import streamlit as st
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# Fix for some Windows OpenMP conflicts (only if needed)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

API_QUERY = "http://127.0.0.1:8000/query"
SAMPLE_RATE = 16000
RECORD_SECONDS = 4

st.title("Item Memory Assistant")

def parse_object_from_text(text: str):
    t = text.lower()
    if "watch" in t:
        return "MyWatch"
    if "wallet" in t:
        return "MyWallet"
    if "key" in t or "keys" in t:
        return "MyBikeKeys"
    return None

@st.cache_resource
def load_whisper():
    return WhisperModel("base", device="cpu", compute_type="int8")

def record_audio():
    st.info("Recording... speak now")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()
    path = "query.wav"
    write(path, SAMPLE_RATE, audio)
    return path

def transcribe(path):
    whisper = load_whisper()
    segments, info = whisper.transcribe(path, beam_size=5)
    return "".join(seg.text for seg in segments).strip()

query_text = st.text_input("Type your question (e.g., Where did I last see my watch?)")

col1, col2 = st.columns(2)
with col1:
    if st.button("Search (Text)"):
        obj = parse_object_from_text(query_text) or query_text.strip()
        resp = requests.post(API_QUERY, json={"object_name": obj, "k": 3})
        st.json(resp.json())

with col2:
    if st.button("🎤 Speak"):
        wav = record_audio()
        text = transcribe(wav)
        st.write("You said:", text)
        obj = parse_object_from_text(text)
        if not obj:
            st.error("Could not detect object name. Say watch / wallet / keys.")
        else:
            resp = requests.post(API_QUERY, json={"object_name": obj, "k": 3})
            data = resp.json()
            st.subheader(f"Results for {obj}")
            for r in data["results"]:
                st.write(f"Time: {r['time_iso']} | Location: {r.get('location','')}")
                img_path = r.get("image_path")
                if img_path and os.path.exists(img_path):
                    st.image(img_path, caption=f"Event id {r['id']}", use_container_width=True)
