import streamlit as st
import librosa
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard
from io import BytesIO

st.set_page_config(page_title="AI DJ Mixer", page_icon="🎧")
st.title("🎧 Python Automix Studio")

# 1. Custom Fade Function (Replaces non-existent library classes)
def apply_fade(audio_segment, sr, duration_sec, fade_type='out'):
    num_samples = int(duration_sec * sr)
    # Ensure we don't try to fade more samples than the segment contains
    num_samples = min(num_samples, len(audio_segment))
    
    # Create a linear fade curve
    curve = np.linspace(0.0, 1.0, num_samples)
    
    if fade_type == 'out':
        curve = curve[::-1]  # Reverse for fade-out
        audio_segment[-num_samples:] *= curve
    else:
        audio_segment[:num_samples] *= curve
    return audio_segment

# 2. Setup UI
file_a = st.file_uploader("Track A (Out)", type=['mp3', 'wav'])
file_b = st.file_uploader("Track B (In)", type=['mp3', 'wav'])
transition_sec = st.slider("Transition Length (seconds)", 2, 10, 5)

if file_a and file_b:
    if st.button("Generate Seamless Mix"):
        with st.spinner("Analyzing beats and matching tempos..."):
            # Load Audio
            y1, sr1 = librosa.load(file_a, sr=44100)
            y2, sr2 = librosa.load(file_b, sr=44100)

            # Analyze BPM
            tempo1, _ = librosa.beat.beat_track(y=y1, sr=sr1)
            tempo2, _ = librosa.beat.beat_track(y=y2, sr=sr2)
            
            # Time Stretch
            stretch_rate = tempo2 / tempo1
            y2_stretched = librosa.effects.time_stretch(y2, rate=float(stretch_rate))

            # Slice
            overlap_samples = int(transition_sec * sr1)
            track_a_main = y1[:-overlap_samples]
            track_a_fade = y1[-overlap_samples:]
            track_b_fade = y2_stretched[:overlap_samples]
            track_b_main = y2_stretched[overlap_samples:]

            # Apply custom fades
            track_a_faded = apply_fade(track_a_fade, sr1, transition_sec, fade_type='out')
            track_b_faded = apply_fade(track_b_fade, sr1, transition_sec, fade_type='in')

            # Sum and concatenate
            combined_transition = track_a_faded + track_b_faded
            final_mix = np.concatenate([track_a_main, combined_transition, track_b_main])

            # Output
            buffer = BytesIO()
            sf.write(buffer, final_mix, sr1, format='WAV')
            st.success("Mix perfected!")
            st.audio(buffer)
