import os
import tempfile
import subprocess
from datetime import datetime

import streamlit as st
import google.generativeai as genai
from textblob import TextBlob
import requests
import whisper
from faster_whisper import WhisperModel


# ---------- FFmpeg discovery ----------
def ensure_ffmpeg_on_path():
    here = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_local = os.path.join(here, "ffmpeg.exe")
    if os.path.exists(ffmpeg_local):
        os.environ["PATH"] = here + os.pathsep + os.environ.get("PATH", "")
        return

    try:
        import imageio_ffmpeg
        ffbin = imageio_ffmpeg.get_ffmpeg_exe()
        os.environ["PATH"] = os.path.dirname(ffbin) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

ensure_ffmpeg_on_path()


# ---------- Gemini config ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)  # type: ignore
    except Exception:
        pass


# ---------- UI ----------
st.set_page_config(page_title="Sales Call Analyzer", page_icon="🎧", layout="wide")
st.title("🎧 Sales Call Analyzer")

with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox(
        "Whisper model",
        options=["tiny", "base", "small", "medium"],
        index=1,  # default = "base"
        help="Larger = better accuracy, slower."
    )
    language_opt = st.selectbox(
        "Language",
        options=["auto", "en", "hi", "es", "fr", "de"],
        index=0,
        help="If your calls are English, pick 'en' for better accuracy."
    )
    run_gemini = st.toggle("Run Gemini analysis", value=True)

    # 🔹 External API input
    external_api_url = st.text_input(
        "🔗 External Analysis API",
        placeholder="Paste API URL here (optional)"
    )

    st.caption("Tip: put ffmpeg.exe next to app.py on Windows if decoding fails.")

    # 🔹 Clear Dashboard button
    if st.button("Clear Dashboard"):
        st.session_state['analysis_results'] = []
        st.session_state['uploaded_files'] = []
        st.rerun()





@st.cache_resource(show_spinner=False)
def load_whisper(model_name: str):
    """Load faster-whisper model (CPU, int8 for low memory use)."""
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def preprocess_to_wav(input_path: str, out_rate=16000, out_channels=1) -> str:
    """Convert any media to 16kHz mono WAV using ffmpeg."""
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(out_rate),
        "-ac", str(out_channels),
        "-f", "wav",
        tmp_wav
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{proc.stderr.decode(errors='ignore')}")
    return tmp_wav


def transcribe(file_path: str, lang: str | None, model: WhisperModel):
    """Run faster-whisper transcription."""
    segments, _info = model.transcribe(file_path, language=None if lang == "auto" else lang)
    text = " ".join([seg.text for seg in segments])
    return text.strip()


def analyze_with_gemini(transcript: str) -> str:
    """Sales-centric analysis with Gemini."""
    if not GEMINI_API_KEY:
        return "Gemini API key not set. Set GEMINI_API_KEY to enable AI insights."

    prompt = f"""
You are a sales call analyst (you're knowledgeable in sales strategies and customer interactions). 
You also know Hindi, Hinglish, and English well.
---
{transcript}
---

Provide:
1) A concise bullet summary
2) Customer sentiment (positive/neutral/negative) and why
3) Key objections or opportunities
4) Action items for the sales rep
5) Strategy improvements for the founder

Keep it short and skimmable.
"""
    try:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")  # type: ignore
            resp = model.generate_content(prompt)
        except Exception:
            resp = genai.generate_content(model="gemini-1.5-flash", prompt=prompt)  # type: ignore
        return getattr(resp, "text", "").strip() or "No response text returned."
    except Exception as e:
        return f"Gemini error: {e}"


def analyze_with_external_api(transcript: str, api_url: str) -> str:
    """Send transcript to user-provided external API."""
    try:
        resp = requests.post(api_url, json={"transcript": transcript}, timeout=30)
        if resp.status_code == 200:
            return resp.text
        return f"External API error: {resp.status_code} {resp.text}"
    except Exception as e:
        return f"External API request failed: {e}"


def text_sentiment_blob(text: str):
    blob = TextBlob(text)
    return blob.sentiment


# ---------- Main Flow ----------
uploaded = st.file_uploader(
    "Upload an audio file (mp3, wav, m4a, aac, flac, ogg…)",
    type=["mp3", "wav", "m4a", "aac", "flac", "ogg", "wma", "webm", "mp4"],
    accept_multiple_files=False
)

if uploaded:
    suffix = os.path.splitext(uploaded.name)[1].lower()
    raw_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
    with open(raw_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.info("Converting to 16kHz mono WAV…")
    try:
        wav_path = preprocess_to_wav(raw_path)
    except Exception as e:
        st.error(f"Audio conversion failed.\n\n{e}")
        st.stop()

    # Load Whisper only when needed
    whisper_model = load_whisper(model_size)

    with st.spinner("Transcribing with Whisper…"):
        transcript = transcribe(wav_path, language_opt, whisper_model)

    if not transcript:
        st.warning("No transcript produced. Try a different model or check audio quality.")
        st.stop()

    # Transcript
    st.subheader("📝 Transcript")
    st.write(transcript)

    # Quick sentiment
    st.subheader("📊 Sentiment (TextBlob)")
    pol = text_sentiment_blob(transcript)
    st.write(
        f"Polarity: **{pol.polarity:.3f}** (−1 negative, +1 positive) • "
        f"Subjectivity: **{pol.subjectivity:.3f}** (0 objective, 1 subjective)"
    )

    # Gemini analysis
    if run_gemini:
        with st.spinner("Analyzing with Gemini…"):
            insights = analyze_with_gemini(transcript)

        st.subheader("🤖 AI Analysis (Gemini)")
        st.write(insights)

    # External API analysis
    if external_api_url:
        with st.spinner("Analyzing with External API…"):
            ext_insights = analyze_with_external_api(transcript, external_api_url)

        st.subheader("🌐 External API Analysis")
        st.write(ext_insights)

    # Downloads
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "⬇️ Download Transcript",
            data=transcript.encode("utf-8"),
            file_name=f"transcript_{ts}.txt",
            mime="text/plain"
        )
    with col2:
        if run_gemini:
            st.download_button(
                "⬇️ Download Analysis (Gemini)",
                data=insights.encode("utf-8"),
                file_name=f"analysis_gemini_{ts}.txt",
                mime="text/plain"
            )
    with col3:
        if external_api_url:
            st.download_button(
                "⬇️ Download Analysis (External API)",
                data=ext_insights.encode("utf-8"),
                file_name=f"analysis_external_{ts}.txt",
                mime="text/plain"
            )
