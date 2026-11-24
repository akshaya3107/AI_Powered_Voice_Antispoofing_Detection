# AI Powered Voice Antispoofing Detection

A deep learning system that classifies a voice recording as **genuine human speech (REAL)** or **spoofed / AI‑generated audio (FAKE)**, such as deepfakes or synthetic voices. It is meant to harden voice‑based authentication and limit misuse of cloned voices.

The project ships two front ends over the same model:

- **Streamlit web app**: upload a file or record from the microphone and get a verdict, acoustic metrics and a waveform.
- **FastAPI REST service**: a `POST` endpoint for integrating detection into other systems.

---

## Repository structure

```
.
├── API/
│   ├── streamlit_app.py          # Streamlit UI (upload / record → verdict)
│   ├── main.py                   # Starts the FastAPI server (uvicorn)
│   ├── requirements.txt          # Python dependencies
│   └── app/
│       ├── app.py                # FastAPI routes
│       └── src/
│           ├── deepfake.py       # Model loading + inference (infa_deepfake)
│           └── artifact/
│               └── ann_human_or_bot/   # Exported TensorFlow SavedModel
├── deepfake-voice-clasification-yamnet-2.ipynb   # Training notebook (Kaggle)
├── Hackathon_ProblemStatement.pdf
└── Project Presentation VOIS Marathon2.0.pptx
```

---

## How it works

### Model

Training happens in `deepfake-voice-clasification-yamnet-2.ipynb` on the
**In‑the‑Wild audio deepfake** dataset (Kaggle) (`real` / `fake` folders).

1. Every clip is loaded with `librosa` as **16 kHz mono** float audio.
2. **YAMNet**, a pretrained audio network from TensorFlow Hub, stays frozen and turns the waveform into one **1024‑d embedding per ~0.48 s frame**.
3. A small classifier is trained on these embeddings, with each frame labelled after its clip. The notebook compares ANN, CNN and LSTM heads. The **ANN** head is the one the app uses.
4. For serving, YAMNet, the ANN and a frame‑averaging layer are chained into one Keras model. That model is exported as a SavedModel to `API/app/src/artifact/ann_human_or_bot`. It takes a raw 1‑D waveform and returns `[P(FAKE), P(REAL)]`.

```
waveform (16 kHz) ──► YAMNet ──► embeddings [frames × 1024] ──► ANN ──► [frames × 2] ──► mean ──► [P(FAKE), P(REAL)]
```

Because YAMNet is baked into the SavedModel, inference does **not** need TensorFlow Hub or internet access.

### Inference flow

`app/src/deepfake.py` loads the SavedModel **once at import time**. `infa_deepfake(path)` then:

1. loads the audio file as 16 kHz mono (`librosa`, so WAV/FLAC/OGG/MP3/M4A all work; MP3/M4A may need `ffmpeg` on the system),
2. runs the `serving_default` signature,
3. returns `(1, "REAL" | "FAKE")` on success or `(0, "<error message>")` on failure.

Both front ends call this one function.

### Streamlit app flow (`API/streamlit_app.py`)

```
Upload file / Record mic
        │
        ▼
process_audio_file()  ── save to temp file
        ├── extract_audio_metadata()  → duration, sample rate, pitch (YIN), RMS energy, waveform
        └── infa_deepfake()           → REAL / FAKE
        │
        ▼
st.session_state.result  ──►  render_results()  → verdict card, metric cards, waveform plot
```

Uploaded and recorded audio follow the same path, so both get a real model prediction.

### REST API flow (`API/app/app.py`)

`POST /deepfake` saves the uploaded file to a temporary directory, calls `infa_deepfake`, deletes the temp files and returns JSON.

---

## Setup

Requires **Python 3.10 – 3.13**.

```bash
git clone https://github.com/akshaya3107/AI_Powered_Voice_Antispoofing_Detection.git
cd AI_Powered_Voice_Antispoofing_Detection

python -m venv venv
# Windows:      venv\Scripts\activate
# Linux/macOS:  source venv/bin/activate

cd API
pip install -r requirements.txt
```

All commands below run from the `API/` directory.

### Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Then choose **Upload** or **Record**, give it audio and click **Run analysis**.

### Run the REST API

```bash
python main.py
```

The server listens on http://localhost:8000. Interactive docs are at http://localhost:8000/docs.

| Method | Path        | Description |
|--------|-------------|-------------|
| GET    | `/`         | Health check |
| POST   | `/deepfake` | Classify an audio file (multipart field `audio_file`) |
| POST   | `/depfake`  | Legacy alias of `/deepfake` |

Example:

```bash
curl -X POST -F "audio_file=@sample.wav" http://localhost:8000/deepfake
```

```json
{"status": 1, "Message": "REAL"}
```

On failure, `status` is `0` and `Message` holds the error text.

---

## Retraining the model

1. Open `deepfake-voice-clasification-yamnet-2.ipynb`. It was written for Kaggle and expects the dataset at `/kaggle/input/in-the-wild-audio-deepfake/release_in_the_wild`.
2. Run the notebook to train the heads and export `ann_human_or_bot`. It needs `tensorflow_hub` and internet access to download YAMNet.
3. Replace `API/app/src/artifact/ann_human_or_bot/` with the new export.

---

## Limitations

- The model gives a single label per clip and shows no calibrated confidence score.
- It was trained on one dataset. Accuracy on other recording conditions, languages or newer voice‑cloning systems is not guaranteed.
- The pitch and energy cards describe the acoustics of the clip. They do not identify the speaker.
