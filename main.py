import os
import io
import json
import time
import shutil
import datetime as dt
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

# --------- Load keys from env or Streamlit secrets ----------
OPENAI_API_KEY = "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"
ELEVENLABS_API_KEY = "sk_26e4f15194758ac7f0451cfc08440d035c0fb567bfeb4f32"

# --- Safety checks (simple, friendly errors) ---
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Set it in your environment or Streamlit secrets.")
if not ELEVENLABS_API_KEY:
    st.error("Missing ELEVENLABS_API_KEY. Set it in your environment or Streamlit secrets.")

# --------- OpenAI + ElevenLabs clients ----------
from openai import OpenAI as OpenAIClient
from elevenlabs.client import ElevenLabs

openai_client = OpenAIClient(api_key=OPENAI_API_KEY)
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# ----------------- Page config -----------------
st.set_page_config(page_title="Prompt Playground", layout="wide")

# ----------------- Helpers -----------------
def now_ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_session():
    """Initialize all session_state fields we need."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"ai_{now_ts()}"
    if "messages" not in st.session_state:
        # message: {"role": "user"|"assistant", "content": str, "audio_path": Optional[str], "ts": iso}
        st.session_state.messages: List[Dict[str, Any]] = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = "You are a helpful assistant."
    if "save_conv" not in st.session_state:
        st.session_state.save_conv = True  # default on per your flow
    if "voice_id" not in st.session_state:
        st.session_state.voice_id = "21m00Tcm4TlvDq8ikWAM"  # default example
    if "model_id" not in st.session_state:
        st.session_state.model_id = "gpt-4o-mini"
    if "tts_model" not in st.session_state:
        st.session_state.tts_model = "eleven_multilingual_v2"
    if "audio_format" not in st.session_state:
        st.session_state.audio_format = "mp3_44100_128"
    if "pending_zip" not in st.session_state:
        st.session_state.pending_zip = None  # path to zip ready to download

def session_folder() -> Path:
    base = Path("session_audios")
    base.mkdir(exist_ok=True, parents=True)
    cur = base / st.session_state.session_id
    cur.mkdir(exist_ok=True, parents=True)
    return cur

def write_transcript_json():
    """Write/overwrite transcript JSON for the current session."""
    folder = session_folder()
    transcript = {
        "session_id": st.session_state.session_id,
        "created_at": dt.datetime.now().isoformat(),
        "model": st.session_state.model_id,
        "tts_model": st.session_state.tts_model,
        "voice_id": st.session_state.voice_id,
        "messages": st.session_state.messages,
    }
    with open(folder / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

def zip_current_session() -> Path:
    """Zip the current session folder and return zip path."""
    folder = session_folder()
    zip_name = folder.with_suffix("")  # strip . if any
    zip_path = shutil.make_archive(str(zip_name), "zip", root_dir=str(folder))
    return Path(zip_path)

def reset_for_new_session():
    st.session_state.pending_zip = None
    st.session_state.session_id = f"ai_{now_ts()}"
    st.session_state.messages = []
    # Send a hidden starter message to kick off AI intro
    st.session_state.messages.append({
        "role": "user",
        "content": "Start a new conversation.",
        "audio_path": None,
        "ts": dt.datetime.now().isoformat(),
    })
    print("Starting Message:", st.session_state.messages)
    response_text = get_openai_response(messages=st.session_state.messages)

    # Generate AI audio
    audio_bytes = elevenlabs_tts(response_text)
    msg_index = 1  # first AI message in session
    save_ai_message(response_text, audio_bytes, msg_index)

def render_chat_bubble(msg: Dict[str, Any], idx: int):
    role = msg.get("role", "assistant")
    text = msg.get("content", "")
    audio_path = msg.get("audio_path")

    # colors: same family, different shade
    if role == "assistant":
        align = "flex-start"
        bg = "#e8f0ff"   # lighter shade
    else:
        align = "flex-end"
        bg = "#cfe0ff"   # darker shade of same family

    st.markdown(
        f"""
        <div style="
            display:flex;
            justify-content:{ 'flex-start' if role=='assistant' else 'flex-end' };
            margin: 8px 0;">
            <div style="
                max-width: 85%;
                background: {bg};
                color: #0b2545;
                padding: 12px 14px;
                border-radius: 12px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.06);
                ">
                <div style="font-size: 0.85rem; opacity: 0.7; margin-bottom: 6px;">
                    {"AI" if role=="assistant" else "You"}
                </div>
                <div style="white-space: pre-wrap; font-size: 1rem;">{text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if role == "assistant" and audio_path and Path(audio_path).exists():
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/mp3")

def save_ai_message(text: str, audio_bytes: bytes, msg_index: int) -> str:
    """Save AI audio and append to messages with relative path."""
    folder = session_folder()
    audio_name = f"assistant_msg_{msg_index}.mp3"
    audio_path = folder / audio_name
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    # Store relative path for portability
    rel_path = str(audio_path.as_posix())
    st.session_state.messages.append({
        "role": "assistant",
        "content": text,
        "audio_path": rel_path,
        "ts": dt.datetime.now().isoformat(),
    })
    if st.session_state.save_conv:
        write_transcript_json()
    return rel_path

def save_user_message(text: str):
    st.session_state.messages.append({
        "role": "user",
        "content": text,
        "audio_path": None,
        "ts": dt.datetime.now().isoformat(),
    })
    if st.session_state.save_conv:
        write_transcript_json()

# =============== UI ===============
ensure_session()

def get_openai_response(system_prompt: str = st.session_state.system_prompt, messages: List[Dict[str, str]] = None) -> str:
    """
    Use the Responses API with the user's provided snippet style.
    We convert our message history into the simple role/content list.
    """
    # Build the list: include system as a "system" role item if provided
    input_list = []
    if system_prompt.strip():
        input_list.append({"role": "system", "content": system_prompt.strip()})
    for m in messages:
        input_list.append({"role": m["role"], "content": m["content"]})

    start = time.time()
    resp = openai_client.responses.create(
        model=st.session_state.model_id,
        input=input_list
    )
    response_time = time.time() - start
    st.write(f"Response time: {response_time:.2f} seconds")
    # The sample they gave uses response.output_text
    return resp.output_text

def elevenlabs_tts(text: str) -> bytes:
    """
    Generate audio via ElevenLabs and return bytes.
    """
    audio_stream = eleven_client.text_to_speech.convert(
        text=text,
        voice_id=st.session_state.voice_id,
        model_id=st.session_state.tts_model,
        output_format=st.session_state.audio_format,
        voice_settings={
            "stability": st.session_state.voice_stability,
            "similarity_boost": st.session_state.voice_similarity,
            "style": st.session_state.voice_style,
            "speed": st.session_state.voice_speed,
        }

    )
    # convert generator to bytes
    audio_bytes = b"".join(audio_stream)
    return audio_bytes

# --- Top controls row ---
col_left, col_mid, col_right = st.columns([1.0, 1.6, 1.0])

with col_left:
    st.subheader("System Prompt")
    st.session_state.system_prompt = st.text_area(
        label="",
        value=st.session_state.system_prompt,
        height=220,
        placeholder="Set the system behavior here...",
    )

    st.markdown("---")
    st.caption("Session")
    st.write(f"**ID**: {st.session_state.session_id}")

    st.session_state.save_conv = st.toggle("Save conversation", value=st.session_state.save_conv)

    if st.button("🆕 Start New Conversation"):
        # When starting new: zip existing (if any content), offer download, then reset
        if st.session_state.messages:
            # zip and hold
            zip_path = zip_current_session()
            st.session_state.pending_zip = str(zip_path)
        reset_for_new_session()
        st.rerun()

with col_right:
    st.subheader("Extra Fields")
    with st.expander("Coming soon"):
        st.text_input("Custom field 1")
        st.text_input("Custom field 2")
        st.text_area("Notes")

    st.markdown("---")
    st.caption("Models")
    st.text_input("OpenAI Model", value=st.session_state.model_id, key="model_id")
    st.text_input("ElevenLabs Voice ID", value=st.session_state.voice_id, key="voice_id")
    st.text_input("ElevenLabs TTS Model", value=st.session_state.tts_model, key="tts_model")
    st.selectbox("Audio format", ["mp3_44100_128", "mp3_44100_64", "pcm_16000"], index=0, key="audio_format")

    st.markdown("---")
    st.caption("Voice Settings")
    st.session_state.voice_stability = st.slider(
        "Stability (0 = very expressive, 1 = very consistent)", 0.0, 1.0, 0.30, 0.05
    )
    st.session_state.voice_similarity = st.slider(
        "Similarity Boost (0 = loose, 1 = strong)", 0.0, 1.0, 0.30, 0.05
    )
    st.session_state.voice_style = st.slider(
        "Style Strength (0 = natural, 1 = stylized)", 0.0, 1.0, 0.03, 0.01
    )
    st.session_state.voice_speed = st.slider(
        "Speech Speed (0.7x – 1.2x)", 0.7, 1.2, 0.7, 0.05
    )

with col_mid:
    st.subheader("Chat")

    # Show "download previous zip" if present (immediately after user clicks New Conversation)
    if st.session_state.pending_zip:
        try:
            with open(st.session_state.pending_zip, "rb") as f:
                zip_bytes = f.read()
            st.success("Your previous conversation has been archived.")
            st.download_button(
                label="⬇️ Download conversation ZIP",
                data=zip_bytes,
                file_name=Path(st.session_state.pending_zip).name,
                mime="application/zip",
            )
        except Exception as e:
            st.warning(f"Could not provide ZIP automatically. Folder is saved at: {Path(st.session_state.pending_zip).parent}")
        # Once shown, clear so it doesn't persist forever
        st.session_state.pending_zip = None

    # Render history
    for i, msg in enumerate(st.session_state.messages):
        render_chat_bubble(msg, i)

    # Chat input (only in middle column)
    prompt = st.chat_input("Type your message…")

    if prompt:
        # Immediately display the user bubble
        save_user_message(prompt)
        render_chat_bubble(st.session_state.messages[-1], len(st.session_state.messages)-1)

        # Status placeholders: "Thinking..." then "Generating audio..."
        status_box = st.empty()
        status_box.info("Thinking…")

        # 1) Get text from OpenAI
        try:
            ai_text = get_openai_response(
                st.session_state.system_prompt,
                st.session_state.messages  # includes the latest user msg
            )
        except Exception as e:
            status_box.error(f"OpenAI error: {e}")
            st.stop()

        # Update status
        status_box.warning("Generating audio…")

        # 2) Generate audio with ElevenLabs
        try:
            audio_bytes = elevenlabs_tts(ai_text)
        except Exception as e:
            status_box.error(f"ElevenLabs error: {e}")
            # Still save the text response (without audio)
            st.session_state.messages.append({
                "role": "assistant",
                "content": ai_text,
                "audio_path": None,
                "ts": dt.datetime.now().isoformat(),
            })
            if st.session_state.save_conv:
                write_transcript_json()
            render_chat_bubble(st.session_state.messages[-1], len(st.session_state.messages)-1)
            st.stop()

        # 3) Save + render
        msg_index = sum(1 for m in st.session_state.messages if m["role"] == "assistant")
        audio_path = save_ai_message(ai_text, audio_bytes, msg_index)
        status_box.empty()
        render_chat_bubble(st.session_state.messages[-1], len(st.session_state.messages)-1)
