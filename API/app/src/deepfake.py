"""Deepfake voice inference.

Pipeline (baked into the exported SavedModel, see the training notebook):

    16 kHz mono waveform
      -> YAMNet (pretrained, frozen) -> 1024-d embedding per ~0.48 s frame
      -> ANN classifier             -> [P(FAKE), P(REAL)] per frame
      -> mean over frames           -> final [P(FAKE), P(REAL)]
"""
import os

import librosa
import numpy as np
import tensorflow as tf

CLASSES = ["FAKE", "REAL"]
TARGET_SR = 16000

# Load the exported serving model once at import time
current_direction = os.path.dirname(os.path.abspath(__file__))
deepfake_model = tf.saved_model.load(
    os.path.join(current_direction, "artifact", "ann_human_or_bot")
)
_infer = deepfake_model.signatures["serving_default"]


def load_wav_16k_mono(filename):
    """Load an audio file as a float32 mono waveform resampled to 16 kHz."""
    sound_sample, _ = librosa.load(filename, sr=TARGET_SR, mono=True)
    return sound_sample.astype(np.float32)


def infa_deepfake(audio_path):
    """
    Classify an audio file as FAKE or REAL.

    Returns (status, message): status is 1 on success with message the
    predicted label, or 0 on failure with message a string describing the error.
    """
    try:
        wav = load_wav_16k_mono(audio_path)
        if wav.size == 0:
            return 0, "Audio file is empty."

        output = _infer(tf.convert_to_tensor(wav, dtype=tf.float32))
        predictions = output["output_0"].numpy()

        return 1, CLASSES[int(np.argmax(predictions))]

    except Exception as e:
        print(f"infa_deepfake failed: {e}")
        return 0, str(e)
