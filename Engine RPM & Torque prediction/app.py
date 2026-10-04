import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# Set page title and layout
st.set_page_config(page_title="Engine Audio Diagnostics", layout="wide")

st.title("🚗 Real-Time Engine Torque & RPM Predictor")
st.write("Upload a raw multi-channel engine `.wav` file to predict RPM and Torque over 2-second windows using your trained Extra Trees models.")

# Constants matching the notebook preprocessing pipeline
CHUNK_DURATION = 2          # seconds
TARGET_SR = 16000            # resample engine audio to 16 kHz
N_MFCC = 13

# Helper to load models
@st.cache_resource
def load_models():
    pkl_filename = 'extra_trees_models.pkl'
    if not os.path.exists(pkl_filename):
        st.error(f"Could not find '{pkl_filename}' in the current directory. Please make sure you place the saved pickle file in the same folder as this script.")
        return None
    with open(pkl_filename, 'rb') as f:
        data = pickle.load(f)
    return data

models_data = load_models()

if models_data is not None:
    rpm_model = models_data['rpm_model']
    torque_model = models_data['torque_model']
    feature_names = models_data.get('feature_names', [])
    
    st.success("Models loaded successfully!")

    # Audio File Uploader
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

    if uploaded_file is not None:
        with st.spinner("Processing audio file and extracting features..."):
            try:
                # Load the complete audio
                audio, sr = sf.read(uploaded_file)
                
                # Handle cases where the dataset format differs (checking for 4 channels)
                if audio.ndim == 2 and audio.shape[1] == 4:
                    st.info("Detected standard 4-channel simulation recording. Using Channel 1-2 (stereo engine sound) for predictions.")
                    engine_mono = audio[:, :2].mean(axis=1)
                else:
                    st.warning(f"Audio does not have 4 channels (found {audio.ndim}D shape: {audio.shape}). Treating entire file as raw engine sound (mono). ")
                    if audio.ndim == 2:
                        engine_mono = audio.mean(axis=1)
                    else:
                        engine_mono = audio

                # Chunking and feature extraction
                chunk_samples = int(CHUNK_DURATION * sr)
                num_chunks = len(engine_mono) // chunk_samples
                
                if num_chunks == 0:
                    st.error("Audio file is too short! It must be at least 2 seconds long.")
                else:
                    X_features = []
                    timestamps = []

                    for i in range(num_chunks):
                        start = i * chunk_samples
                        end = start + chunk_samples
                        
                        audio_chunk = engine_mono[start:end]
                        timestamps.append(i * CHUNK_DURATION)
                        
                        # Resample audio to match trained configuration (16kHz)
                        audio_16k = librosa.resample(audio_chunk, orig_sr=sr, target_sr=TARGET_SR)
                        
                        # 1. MFCC features
                        mfcc = librosa.feature.mfcc(y=audio_16k, sr=TARGET_SR, n_mfcc=N_MFCC)
                        mfcc_mean = mfcc.mean(axis=1)
                        mfcc_std = mfcc.std(axis=1)
                        
                        # 2. Spectral features
                        spectral_centroid = librosa.feature.spectral_centroid(y=audio_16k, sr=TARGET_SR)
                        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_16k, sr=TARGET_SR)
                        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_16k, sr=TARGET_SR)
                        zero_crossing = librosa.feature.zero_crossing_rate(audio_16k)
                        rms = librosa.feature.rms(y=audio_16k)
                        
                        extra_features = np.array([
                            spectral_centroid.mean(), spectral_centroid.std(),
                            spectral_bandwidth.mean(), spectral_bandwidth.std(),
                            spectral_rolloff.mean(), spectral_rolloff.std(),
                            zero_crossing.mean(), zero_crossing.std(),
                            rms.mean(), rms.std()
                        ])
                        
                        # Combine features
                        combined_features = np.concatenate([mfcc_mean, mfcc_std, extra_features])
                        X_features.append(combined_features)

                    X_features = np.array(X_features)
                    
                    # Run predictions!
                    predicted_rpm = rpm_model.predict(X_features)
                    predicted_torque = torque_model.predict(X_features)
                    
                    # Create Results DataFrame
                    results_df = pd.DataFrame({
                        'Time (s)': timestamps,
                        'Predicted RPM': predicted_rpm,
                        'Predicted Torque (Nm)': predicted_torque
                    })
                    
                    # Metrics columns
                    st.subheader("⚡ Dynamic Diagnostics Summary")
                    col1, col2, col3 = st.columns(3)
                    col1.metric(label="Max Predicted RPM", value=f"{predicted_rpm.max():.1f} RPM")
                    col2.metric(label="Max Predicted Torque", value=f"{predicted_torque.max():.1f} Nm")
                    col3.metric(label="Total Analysis Duration", value=f"{num_chunks * CHUNK_DURATION} seconds")
                    
                    # Plots
                    st.subheader("📊 Diagnostic Graphs")
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                    
                    ax1.plot(results_df['Time (s)'], results_df['Predicted RPM'], color='tab:red', linewidth=2, marker='o')
                    ax1.set_ylabel('RPM', color='tab:red')
                    ax1.grid(True, linestyle='--', alpha=0.5)
                    ax1.set_title("Predicted RPM over Time")
                    
                    ax2.plot(results_df['Time (s)'], results_df['Predicted Torque (Nm)'], color='tab:blue', linewidth=2, marker='o')
                    ax2.set_ylabel('Torque (Nm)', color='tab:blue')
                    ax2.set_xlabel('Time (seconds)')
                    ax2.grid(True, linestyle='--', alpha=0.5)
                    ax2.set_title("Predicted Torque over Time")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Table display
                    st.subheader("📋 Detailed Predictions Table")
                    st.dataframe(results_df.style.format({'Predicted RPM': '{:.1f}', 'Predicted Torque (Nm)': '{:.2f}'}))

            except Exception as e:
                st.error(f"Error processing the uploaded audio file: {e}")
