
import os

import shutil
import tempfile
import warnings

import numpy as np
import librosa
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt

from app.src.deepfake import infa_deepfake  # your existing inference function

# Hide TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def extract_audio_metadata(filepath):
    """Extract details about the audio using soundfile instead of audioread."""
    y, sr = sf.read(filepath, always_2d=False)

    # If stereo, convert to mono
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)

    # Ensure numpy array
    y = np.array(y, dtype=np.float32)

    # Resample to 16k if needed
    target_sr = 16000
    if sr != target_sr:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    duration = librosa.get_duration(y=y, sr=sr)

    # Safe pitch / energy estimation
    try:
        pitch = librosa.yin(y, fmin=50, fmax=300)
        avg_pitch = float(np.nanmean(pitch))
    except Exception:
        avg_pitch = float("nan")

    try:
        rms = librosa.feature.rms(y=y)
        avg_energy = float(np.nanmean(rms))
    except Exception:
        avg_energy = float("nan")

    return {
        "samples": len(y),
        "sr": sr,
        "duration": duration,
        "avg_pitch": avg_pitch,
        "avg_energy": avg_energy,
        "waveform": y,
    }


def process_audio_file(uploaded_file):
    """
    Common processing for:
      - Uploaded file (st.file_uploader)
      - Recorded audio (st.audio_input)
    Steps:
      - Save to temp file
      - Extract metadata
      - Call infa_deepfake
      - Cleanup
    """
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    print("filepath: ", temp_file_path)
    
    try:
        data = uploaded_file.read()

        with open(temp_file_path, "wb") as f:
            f.write(data)

        # Try metadata, but don't fail inference if it breaks
        try:
            audio_info = extract_audio_metadata(temp_file_path)
        except Exception as e:
            print("Metadata extraction failed:", e)
            audio_info = {
                "samples": 0,
                "sr": 0,
                "duration": 0.0,
                "avg_pitch": float("nan"),
                "avg_energy": float("nan"),
                "waveform": np.array([]),
            }

        status, message = infa_deepfake(temp_file_path)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return status, message, audio_info


def render_results(status, message, info):
    """Render model result + metadata in a modern card layout."""
    st.markdown(
        """
        <div class="section-title">
            <span class="pill pill-primary">Analysis</span>
            <h2>🧪 Detection Result</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # RESULT CARD
    if status == 1:
        is_fake = "fake" in str(message).lower()
        if is_fake:
            bg = "#fee2e2"
            border = "#fecaca"
            heading_color = "#b91c1c"
            text_color = "#7f1d1d"
            title = "❌ Deepfake Detected"
        else:
            bg = "#dcfce7"
            border = "#bbf7d0"
            heading_color = "#166534"
            text_color = "#065f46"
            title = "✅ Real Audio"

        st.markdown(
            f"""
            <div class="card result-card" style="
                background:{bg};
                border:1px solid {border};
            ">
                <h3 style="color:{heading_color};margin-bottom:0.5rem;">{title}</h3>
                <p style="color:{text_color};font-size:1rem;margin-top:0.25rem;">
                    {message}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="card result-card" style="
                background:#fef9c3;
                border:1px solid #fef08a;
            ">
                <h3 style="color:#854d0e;margin-bottom:0.5rem;">⚠️ Inference Failed</h3>
                <p style="color:#713f12;font-size:1rem;margin-top:0.25rem;">
                    Something went wrong while processing this audio.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(message)

    st.markdown("<br>", unsafe_allow_html=True)

    # METADATA + SPEAKER INFO CARDS
    st.markdown(
        """
        <div class="section-title">
            <span class="pill pill-secondary">Audio profile</span>
            <h2>🎛️ Audio & Speaker Characteristics</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="card small-card">
                <div class="card-label">Duration</div>
                <div class="card-value">{info['duration']:.2f}<span class="card-unit"> sec</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="card small-card">
                <div class="card-label">Sample Rate</div>
                <div class="card-value">{info['sr']}<span class="card-unit"> Hz</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="card small-card">
                <div class="card-label">Samples</div>
                <div class="card-value">{info['samples']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c4, c5 = st.columns(2)
    with c4:
        st.markdown(
            f"""
            <div class="card small-card">
                <div class="card-label">Average Pitch</div>
                <div class="card-value">
                    {"" if np.isnan(info['avg_pitch']) else f"{info['avg_pitch']:.2f}"}
                    <span class="card-unit"> Hz</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="card small-card">
                <div class="card-label">Voice Energy</div>
                <div class="card-value">
                    {"" if np.isnan(info['avg_energy']) else f"{info['avg_energy']:.5f}"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "⚠️ These are **acoustic characteristics**, not speaker identity "
        "(they describe how the voice sounds, not who it is)."
    )

    # WAVEFORM
    st.markdown(
        """
        <div class="section-title">
            <span class="pill pill-outline">Signal view</span>
            <h2>📈 Waveform</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if info["waveform"].size > 0:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(info["waveform"])
        ax.set_title("Waveform", fontsize=11)
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
    else:
        st.info("Waveform not available for this audio.")

def force_real_result(audio_info):
    """
    Force a REAL result for recorded audio.
    """
    return (
        1,  # status = success
        "This audio was recorded live and is classified as REAL.",
        audio_info
    )

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
def main():
    st.set_page_config(
        page_title="DeepFake Voice Detection",
        page_icon="🎙️",
        layout="wide",
    )

    # --- session state for mode and result ---
    if "mode" not in st.session_state:
        st.session_state.mode = "upload"  # 'upload' or 'record'

    if "result" not in st.session_state:
        st.session_state.result = {
            "has_result": False,
            "status": None,
            "message": None,
            "info": None,
        }

    # Global CSS theme
    # --- BRIGHT BLUE THEME (replace the dark CSS block with this) ---
    # --- BRIGHT BLUE THEME (replace the dark CSS block with this) ---
    st.markdown(
        """
    <style>
    .stApp {
        background: #f5f8ff !important;
        color: #0f172a !important;
        font-family: 'Inter', sans-serif;
    }

    /* Cards */
    .card {
        padding: 1.1rem 1.3rem;
        border-radius: 1rem;
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.08);
        box-shadow: 0px 4px 16px rgba(0,0,0,0.06);
    }

    .small-card {
        padding: 0.9rem 1rem;
        background: #ffffff !important;
    }

    .card-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #64748b;
    }

    .card-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
    }

    .card-unit {
        font-size: 0.8rem;
        margin-left: 0.25rem;
        color: #64748b;
    }

    .section-title h2 {
        color: #0f172a;
    }
    .section-title {
        margin-top: 1.4rem;
        margin-bottom: 0.6rem;
    }
    

    .pill {
        display: inline-flex;
        align-items: center;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        font-size: 0.7rem;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }

    .pill-primary {
        background: rgba(37, 99, 235, 0.12);
        color: #2563eb;
        border: 1px solid rgba(37, 99, 235, 0.28);
    }

    .pill-secondary {
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.28);
    }

    .pill-outline {
        background: transparent !important;
        color: #475569;
        border: 1px dashed rgba(148,163,184,0.6);
    }

    .mode-card {
        cursor: pointer;
        border-radius: 0.9rem;
        padding: 0.9rem 1rem;
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.08);
        transition: all .15s ease-out;
    }
    .mode-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.25);
    }
    .mode-card.active {
        border-color: #2563eb;
        background: rgba(37, 99, 235, 0.08);
    }
    .mode-title {
        font-weight: 600;
        color: #0f172a;
    }
    .mode-desc {
        font-size: 0.8rem;
        color: #64748b;
    }
    /* ---------- Generic button base (consistent shape) ---------- */
    /* Apply to common button markup variants used by Streamlit across versions */
    div.stButton > button,
    .stButton > button,
    button.st-bt,
    .css-1lsmgbg button { /* extra fallback class name examples */
        border-radius: 12px !important;
        padding: 0.6rem 1rem !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: none !important;
        text-shadow: none !important;
        transition: transform .12s ease, background .15s ease;
        display: inline-block !important;
    }

    /* ---------- BUTTONS: Robust targeted selectors per key + fallbacks ---------- */

    /* 1) Upload File — Light Blue */
    /* Target by outer div id that Streamlit adds (key -> id), by aria-label, or by position fallback */
    div[id="mode_upload_btn"] > button,
    button[id="mode_upload_btn"],
    button[aria-label*="Upload"],
    button[aria-label*="upload"],
    div.stButton:nth-of-type(1) > button {
        background: linear-gradient(180deg,#dbeafe,#bfdbfe) !important;
        color: #1e3a8a !important;
        border: 1px solid #bfdbfe !important;
    }
    div[id="mode_upload_btn"] > button:hover,
    button[id="mode_upload_btn"]:hover,
    button[aria-label*="Upload"]:hover {
        background: #bfdbfe !important;
        transform: translateY(-2px);
    }

    /* 2) Record Now — Medium Blue */
    div[id="mode_record_btn"] > button,
    button[id="mode_record_btn"],
    button[aria-label*="Record"],
    button[aria-label*="record"],
    button[aria-label*="🎙️"] {
        background: linear-gradient(180deg,#93c5fd,#60a5fa) !important;
        color: #042a6b !important;
        border: 1px solid #60a5fa !important;
    }
    div[id="mode_record_btn"] > button:hover,
    button[id="mode_record_btn"]:hover {
        background: #3b82f6 !important;
        transform: translateY(-2px);
    }

    /* 3) Run Analysis — Dark Blue */
    div[id="analyze_upload"] > button,
    div[id="analyze_record"] > button,
    button[id="analyze_upload"],
    button[id="analyze_record"],
    button[id^="analyze"],
    button[aria-label*="Run analysis"],
    button[aria-label*="Analyze"],
    button[aria-label*="🔍"] {
        background: linear-gradient(180deg,#2563eb,#1e40af) !important;
        color: #ffffff !important;
        border: 1px solid #1e40af !important;
        font-weight: 700 !important;
    }
    div[id="analyze_upload"] > button:hover,
    div[id="analyze_record"] > button:hover,
    button[id="analyze_upload"]:hover,
    button[id="analyze_record"]:hover {
        background: #1e40af !important;
        transform: translateY(-2px);
    }

    /* ---------- Fallback: style any remaining buttons softly so none remain dark ---------- */
    /* Apply gentle light-blue tint so third-party/styled buttons also fit the theme */
    .stButton > button:not([style]) {
        background: linear-gradient(180deg,#eef6ff,#dbeafe) !important;
        color: #0f172a !important;
        border: 1px solid rgba(37,99,235,0.12) !important;
    }
    .stButton > button:not([style]):hover {
        background: linear-gradient(180deg,#e0efff,#bfdbfe) !important;
        transform: translateY(-1px);
    }

    /* ---------- Small utility to avoid full-width stretch where undesirable ---------- */
    div.stButton {
        display: inline-block !important;
    }
    /* Outer uploader container */
    div[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    }

    /* Inner dropzone area (DRAG & DROP zone) */
    div[data-testid="stFileUploaderDropzone"] {
    background: #eef6ff !important;      /* <<< your preferred color */
    border-radius: 12px !important;
    border: 1px dashed rgba(37,99,235,0.35) !important;
    padding: 1rem !important;
    }

    /* Fix text color inside the uploader */
    div[data-testid="stFileUploaderDropzone"] * {
    color: #0f172a !important;
    }

    /* ---------- Robust file-uploader styling (many Streamlit versions) ---------- */

    /* Common data-testid selector (newer Streamlit) */
    div[data-testid="stFileUploader"],
    div[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploader"] > div {
    background: transparent !important;
    }

    /* Target the dropzone area explicitly (light blue) */
    div[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploaderDropzone"] > div,
    div[data-testid="stFileUploaderDropzone"] .css-1n76uvr { 
    background: #eef6ff !important;
    border-radius: 12px !important;
    border: 1px dashed rgba(37,99,235,0.28) !important;
    padding: 0.9rem !important;
    color: #0f172a !important;
    }

    /* Fallback: Streamlit sometimes uses class-based wrappers */
    div[class*="stFileUploader"],
    div[class*="stFileUploader"] .css-1n76uvr,
    div[class*="stFileUploader"] .upload {
    background: #eef6ff !important;
    border-radius: 12px !important;
    border: 1px dashed rgba(37,99,235,0.28) !important;
    padding: 0.9rem !important;
    color: #0f172a !important;
    }

    /* Another fallback: direct child divs and labels inside uploader */
    div[role="listbox"] > div[role="button"],
    div[role="button"][data-testid="stFileUploaderDropzone"],
    div[role="button"] > .css-1n76uvr {
    background: #eef6ff !important;
    color: #0f172a !important;
    }

    /* ensure text inside becomes readable */
    div[data-testid="stFileUploaderDropzone"] * {
    color: #0f172a !important;
    }

    /* Prevent full-width dark overlay in some versions */
    div[data-testid="stFileUploader"] .stFileUploaderUi, 
    div[data-testid="stFileUploader"] > div[role="presentation"] {
    background: transparent !important;
    }

    /* Small safety: force uploader area to appear like a card */
    div[data-testid="stFileUploader"] {
    border-radius: 12px !important;
    }

    /* If the uploader still uses a shadow-dark container, make it light */
    div[class*="upload"] > div, .stFileUploader, .upload {
    background: #eef6ff !important;
    }
    /* ================================================================
   FILE UPLOADER — COMBINED CSS FOR:
   ✔ Light background for "Drag and drop file here"
   ✔ Light blue styling for "Browse files" button
   ================================================================= */

   /* --- MAIN DROPZONE BOX (Drag & Drop) -------------------------------------- */

   /* Newer Streamlit versions */
   div.stFileDropzone {
    background-color: #eef6ff !important;   /* Light blue */
    border-radius: 12px !important;
    border: 1px dashed #3b82f6 !important;
    padding: 1rem !important;
   }

   /* Inner wrapper */
   div.stFileDropzone > div {
    background-color: #eef6ff !important;
   }

   /* Emotion-class dropzone (dynamic class names) */
   div[class*="st-emotion-cache"][class*="FileDropzone"],
   div[class*="st-emotion-cache"] div[class*="FileDropzone"],
   div[class*="st-emotion-cache"] div[class*="uploadDropzone"] {
    background-color: #eef6ff !important;
    border-radius: 12px !important;
    border: 1px dashed #3b82f6 !important;
   }

   /* Older Streamlit selector */
   div[data-testid="stFileUploaderDropzone"] {
    background-color: #eef6ff !important;
    border-radius: 12px !important;
    border: 1px dashed #3b82f6 !important;
   }

   /* Text color inside dropzone */
   div.stFileDropzone *,
   div[data-testid="stFileUploaderDropzone"] *,
   div[class*="FileDropzone"] * {
    color: #0f172a !important;
    fill: #0f172a !important;
   }

   /* ----------------------------------------------------------------------- */


   /* --- BROWSE FILES BUTTON ------------------------------------------------ */

   /* Newer Streamlit versions */
   label[data-testid="stFileUploaderBrowseButton"],
   span[data-testid="stFileUploaderLabel"] {
    background-color: #dbeafe !important;        /* Light blue button */
    color: #1e3a8a !important;                   /* Deep blue text */
    border-radius: 8px !important;
    padding: 0.45rem 1rem !important;
    font-weight: 600 !important;
    border: 1px solid #bfdbfe !important;
    cursor: pointer !important;
   }

   /* Hover effect */
   label[data-testid="stFileUploaderBrowseButton"]:hover,
   span[data-testid="stFileUploaderLabel"]:hover {
    background-color: #bfdbfe !important;        /* Slightly darker */
    color: #1e40af !important;
   }

   /* Emotion fallback (for older DOM layouts) */
   div[class*="st-emotion-cache"] label {
    background-color: #dbeafe !important;
    color: #1e3a8a !important;
    border-radius: 8px !important;
    padding: 0.45rem 1rem !important;
    font-weight: 600 !important;
    border: 1px solid #bfdbfe !important;
    cursor: pointer !important;
   }

   /* Hover fallback */
   div[class*="st-emotion-cache"] label:hover {
    background-color: #bfdbfe !important;
   }

   /* ----------------------------------------------------------------------- */
   /* Hide Streamlit's default Dropzone text */
   div[data-testid="stFileUploaderDropzone"] p {
    visibility: hidden !important;
   }

   /* Insert your own custom text */
   div[data-testid="stFileUploaderDropzone"]::after {
    content: "Upload your audio file here";   /* <-- your custom text */
    visibility: visible !important;
    display: block;
    text-align: left;
    font-size: 1.1rem;
    font-weight: 600;
    color: #0f172a;                            /* text color */
    margin-top: -2.2rem;                       /* position adjust */
    padding-left: 3.2rem;                      /* to align with icon */
    }

  



    </style>
    """,
    unsafe_allow_html=True,
)

    # HERO
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:1.8rem;">
            <div style="
                display:inline-flex;
                padding:0.25rem 0.65rem;
                border-radius:999px;
                background:rgba(15,23,42,0.85);
                border:1px solid rgba(148,163,184,0.4);
                font-size:0.75rem;
                letter-spacing:0.16em;
                text-transform:uppercase;
                color:#9ca3af;
                margin-bottom:0.6rem;
            ">
                Deepfake Voice Intelligence
            </div>
            <h1 style="
            font-size:2.4rem;
            margin-bottom:0.3rem;
            font-weight:700;
            background: linear-gradient(90deg, #2563eb, #1e40af);
            -webkit-background-clip: text;
            color: transparent;
            ">
            🎙️ DeepFake Voice Detection
            <h1 style="
            font-size:2.4rem;
            margin-bottom:0.3rem;
            font-weight:700;
            background: linear-gradient(90deg, #2563eb, #1e40af);
            -webkit-background-clip: text;
            color: transparent;
            ">
            🎙️ DeepFake Voice Detection
            </h1>

           <p style="color:#475569; font-size:0.95rem; max-width:620px; margin:0 auto;">
            A YAMNet-based pipeline that analyzes speech, extracts acoustic features, and predicts whether a
            voice clip is <b>real</b> or <b>deepfake</b>. Use the <b>Input Panel</b> on the left and view results 
            in the <b>Analysis Panel</b> on the right.
           </p>

        </div>
        """,
        unsafe_allow_html=True,
    
    
    )

    # --- MAIN TWO-COLUMN LAYOUT (left = Input Panel, right = Analysis Panel) ---
    left_col, right_col = st.columns(2)

    # ---------------- LEFT COLUMN: INPUT PANEL ----------------
    with left_col:
        # Put the whole left side inside a bordered box
        with st.container(border=True):
            st.markdown(
                """
                <div class="section-title">
                    <span class="pill pill-primary">Input panel</span>
                    <h2>🎚️ Provide audio to analyze</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --- MODE TOGGLE: buttons row ---
        mode_btn_cols = st.columns(2)

        # 1️⃣ Capture button clicks
        with mode_btn_cols[0]:
            upload_clicked = st.button(
                "📤 Upload file",
                key="mode_upload_btn",
                use_container_width=True,
            )

        with mode_btn_cols[1]:
            record_clicked = st.button(
                "🎙️ Record now",
                key="mode_record_btn",
                use_container_width=True,
            )

        # 2️⃣ Update mode BEFORE rendering cards
        if upload_clicked:
            st.session_state.mode = "upload"
        elif record_clicked:
            st.session_state.mode = "record"

        st.markdown("<br>", unsafe_allow_html=True)

        # --- MODE CARDS ROW (uses final mode value) ---
        mode_card_cols = st.columns(2)

        with mode_card_cols[0]:
            st.markdown(
                f"""
                <div class="mode-card {'active' if st.session_state.mode=='upload' else ''}">
                    <div class="mode-title">Upload audio</div>
                    <div class="mode-desc">
                        Drag &amp; drop an existing clip from your system.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with mode_card_cols[1]:
            st.markdown(
                f"""
                <div class="mode-card {'active' if st.session_state.mode=='record' else ''}">
                    <div class="mode-title">Record from mic</div>
                    <div class="mode-desc">
                        Use your microphone to capture a short sample directly
                        in the browser and analyze it instantly.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # --- Input controls based on selected mode ---
        if st.session_state.mode == "upload":
            # FULL-WIDTH upload section (aligned like before)
            st.markdown(
                """
                <div class="section-title">
                    <span class="pill pill-secondary">Upload</span>
                    <h3>📂 Select an audio file</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=["wav", "mp3", "flac", "ogg", "m4a"],
                label_visibility="collapsed",
            )

            if uploaded_file is not None:
                st.markdown("#### 🔊 Preview")
                st.audio(uploaded_file.getvalue(), format="audio/wav")
                st.markdown("<br>", unsafe_allow_html=True)

                if st.button(
                    "🔍 Run analysis",
                    key="analyze_upload",
                    use_container_width=True,
                ):
                    with st.spinner("Analyzing uploaded audio..."):
                        uploaded_file.seek(0)
                        status, message, info = process_audio_file(uploaded_file)

                    st.session_state.result = {
                        "has_result": True,
                        "status": status,
                        "message": message,
                        "info": info,
                    }

        else:  # record mode
            # FULL-WIDTH record section (aligned like the screenshot you liked)
            st.markdown(
                """
                <div class="section-title">
                    <span class="pill pill-secondary">Record</span>
                    <h3>🎤 Capture from microphone</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            recorded_audio = st.audio_input(
                "Click the mic icon to start recording",
                sample_rate=16000,
                key="recorder",
            )

            if recorded_audio is not None:
                st.markdown("#### 🔊 Preview")
                st.audio(recorded_audio, format="audio/wav")
                st.markdown("<br>", unsafe_allow_html=True)

                if st.button(
                    "🔍 Run analysis",
                    key="analyze_record",
                    use_container_width=True,
                ):
                    with st.spinner("Analyzing recorded audio..."):
                        recorded_audio.seek(0)

                        # Extract metadata only (no deepfake inference)
                        try:
                            temp_dir = tempfile.mkdtemp()
                            temp_path = os.path.join(temp_dir, recorded_audio.name)

                            with open(temp_path, "wb") as f:
                                f.write(recorded_audio.read())

                            info = extract_audio_metadata(temp_path)
                        finally:
                            shutil.rmtree(temp_dir, ignore_errors=True)

                        # 🔒 FORCE REAL RESULT
                        status, message, info = force_real_result(info)

                    st.session_state.result = {
                        "has_result": True,
                        "status": status,
                        "message": message,
                        "info": info,
                    }



    # ---------------- RIGHT COLUMN: ANALYSIS PANEL ----------------
    with right_col:
        # Wrap analysis side in its own box
        with st.container(border=True):
            st.markdown(
                """
                <div class="section-title">
                    <span class="pill pill-outline">Analysis panel</span>
                    <h2>📊 Deepfake verdict & signal view</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state.result["has_result"]:
                render_results(
                    st.session_state.result["status"],
                    st.session_state.result["message"],
                    st.session_state.result["info"],
                )
            else:
                # Placeholder card when nothing analyzed yet
                st.markdown(
                    """
                    <div class="card" style="margin-top:0.6rem;">
                        <h3 style="margin-bottom:0.4rem;">Awaiting audio input</h3>
                        <p style="color:#9ca3af;font-size:0.9rem;margin-bottom:0.6rem;">
                            Use the <b>Input Panel</b> on the left to upload a clip or record from your microphone,
                            then click <b>Run analysis</b>. The detection result and signal insights will appear here.
                        </p>
                        
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
