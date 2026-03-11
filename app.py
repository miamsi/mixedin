import streamlit as st
import librosa
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, FadeIn, FadeOut
from io import BytesIO

st.set_page_config(page_title="AI DJ Mixer", page_icon="🎧")
st.title("🎧 Python Automix Studio")
st.markdown("Upload two tracks to create a beat-matched transition.")

# 1. Setup UI
col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Track A (Out)", type=['mp3', 'wav'])
with col2:
    file_b = st.file_uploader("Track B (In)", type=['mp3', 'wav'])

transition_sec = st.slider("Transition Length (seconds)", 2, 10, 5)

if file_a and file_b:
    if st.button("Generate Seamless Mix"):
        with st.spinner("Analyzing beats and matching tempos..."):
            
            # 2. Load Audio
            y1, sr1 = librosa.load(file_a, sr=44100)
            y2, sr2 = librosa.load(file_b, sr=44100)

            # 3. Beat Analysis
            tempo1, beats1 = librosa.beat.beat_track(y=y1, sr=sr1)
            tempo2, beats2 = librosa.beat.beat_track(y=y2, sr=sr2)
            
            st.info(f"Detected BPMs: Track A: {int(tempo1)} | Track B: {int(tempo2)}")

            # 4. Time Stretching (Match Track B to Track A's tempo)
            # Formula: rate = Original BPM / Target BPM
            stretch_rate = tempo2 / tempo1
            y2_stretched = librosa.effects.time_stretch(y2, rate=float(stretch_rate))

            # 5. Slice and Dice
            # We take the end of A and the start of B
            overlap_samples = int(transition_sec * sr1)
            
            track_a_main = y1[:-overlap_samples]
            track_a_fade = y1[-overlap_samples:]
            track_b_fade = y2_stretched[:overlap_samples]
            track_b_main = y2_stretched[overlap_samples:]

            # 6. Apply Fades using Pedalboard
            # We convert to float32 for Pedalboard compatibility
            fade_out = FadeOut(duration=transition_sec)(track_a_fade, sr1)
            fade_in = FadeIn(duration=transition_sec)(track_b_fade, sr1)

            # 7. Sum the overlap (The Crossfade)
            combined_transition = fade_out + fade_in

            # 8. Concatenate everything
            final_mix = np.concatenate([track_a_main, combined_transition, track_b_main])

            # 9. Output to Streamlit
            buffer = BytesIO()
            sf.write(buffer, final_mix, sr1, format='WAV')
            st.success("Mix perfected!")
            st.audio(buffer)
